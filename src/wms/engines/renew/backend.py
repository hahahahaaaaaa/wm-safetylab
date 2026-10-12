"""Small complete-support study using the unchanged official RENEW LatentCNN.

This is a controlled adaptation, not full RENEW active querying or paper reproduction.
All heavy imports are restricted to the isolated native process.
"""
from pathlib import Path
import hashlib
import itertools
import json
import math
import sys
import time
import zipfile


def run_pilot(config, case, seed, output, worker):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    from types import SimpleNamespace
    from wms.research.feedback_shift.probability import metrics
    from wms.research.feedback_shift.feedback import make_group, joint_group_channel
    from wms.research.feedback_shift.decision import decide
    from wms.research.feedback_shift import objectives

    output = Path(output)
    source = Path(config['source_dir'])
    sys.path.insert(0, str(source / 'world_models'))
    from network import LatentCNNNetwork, build_network, init_params, save_params, load_params
    from env import make_2048_meta
    from jumanji.environments.logic.game_2048.utils import move
    start = time.monotonic()
    meta = make_2048_meta(board_size=2, max_tile_exp=5)
    env = meta.make_env()
    args = SimpleNamespace(embed_dim=8, latent_channels=8, enc_layers=1, dyn_layers=1,
                           dec_layers=1, beta_bt=1.0, horizon=1, num_candidates=3, exclude_ties=True)
    net = build_network(meta, args)
    support = np.asarray(list(itertools.product(range(5), repeat=4)), dtype=np.int32)
    index = {tuple(row): i for i, row in enumerate(support.tolist())}
    rng_context = np.random.default_rng(seed + 100)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    move_jit = jax.jit(move)
    oracle_batch = jax.jit(jax.vmap(meta.oracle_step, (0, 0, 0)))
    worker.event('progress', {'phase':'collecting_disjoint_contexts', 'support_size':625})
    state, _ = reset(jax.random.PRNGKey(seed + 101))
    boards, seen = [], set()
    # Contexts come from the real environment. Splitting is by board, not by repeated transitions.
    for attempt in range(3000):
        board = np.asarray(state.board, dtype=np.int32).reshape(-1)
        key = tuple(board.tolist())
        valid = np.flatnonzero(np.asarray(state.action_mask))
        if len(valid) and board.max() <= 3:
            action = int(rng_context.choice(valid))
            if key not in seen:
                boards.append({'board':board.tolist(), 'action':action})
                seen.add(key)
                if len(boards) == 40:
                    break
            state, _ = step(state, jnp.int32(action))
        else:
            state, _ = reset(jax.random.PRNGKey(int(rng_context.integers(1,2**30))))
    if len(boards) < 40:
        raise RuntimeError('Not enough distinct environment contexts; no experiment emitted')
    rng_context.shuffle(boards)
    train, test = boards[:24], boards[24:32]
    for split, rows in [('train',train),('evaluation',test)]:
        for i,c in enumerate(rows):
            c.update(case_id=f'{split}-{i:03d}', split=split)

    def enumerate_case(c):
        before=np.asarray(c['board'],dtype=np.int32).reshape(2,2)
        merged,_=move_jit(jnp.asarray(before),jnp.int32(c['action']))
        merged=np.asarray(merged,dtype=np.int32)
        empties=np.flatnonzero(merged.reshape(-1)==0)
        outcomes=[]
        if not np.array_equal(before,merged) and len(empties):
            for pos in empties:
                for exponent,prob in [(1,.9),(2,.1)]:
                    nxt=merged.reshape(-1).copy();nxt[pos]=exponent
                    outcomes.append({'board':nxt.tolist(),'probability':prob/len(empties),
                        'spawn_position':int(pos),'spawn_exponent':exponent,
                        'event':bool(pos==0)})
        else:
            outcomes=[{'board':merged.reshape(-1).tolist(),'probability':1.0,
                       'spawn_position':None,'spawn_exponent':None,'event':False}]
        p=np.zeros(625,dtype=np.float64); event=np.zeros(625,dtype=bool)
        for y in outcomes:
            j=index[tuple(y['board'])];p[j]+=y['probability'];event[j]|=y['event']
        if not np.isclose(p.sum(),1):raise ValueError('Environment enumeration not normalized')
        return p,event,outcomes

    distributions={c['case_id']:enumerate_case(c) for c in train+test}
    # Genuine simulator support checks, separately from the neural model and feedback sampling.
    checked=0
    for ci,c in enumerate(train+test):
        p,_,_=distributions[c['case_id']]
        native_state,_=reset(jax.random.PRNGKey(seed+2000+ci))
        native_state=meta.obs_to_state(jnp.asarray(c['board']),native_state)
        for rep in range(8):
            nxt,_=step(native_state.replace(key=jax.random.PRNGKey(seed+3000+ci*8+rep)),jnp.int32(c['action']))
            if p[index[tuple(np.asarray(nxt.board).reshape(-1).tolist())]]<=0:
                raise ValueError('Actual Jumanji step disagrees with enumerated support')
            checked+=1
    # The unchanged oracle must also preserve no-op boards; they are not impossible outcomes.
    noop={'board':[1,0,2,0],'action':3}
    noop_p,_,_=enumerate_case(noop)
    noop_outputs=oracle_batch(jax.random.split(jax.random.PRNGKey(seed+4000),16),
        jnp.tile(jnp.array(noop['board']), (16,1)),jnp.full((16,),noop['action']))
    if any(noop_p[index[tuple(row)]]<=0 for row in np.asarray(noop_outputs).tolist()):
        raise ValueError('No-op oracle does not match enumerated outcomes')

    obs=jnp.asarray([c['board'] for c in train],dtype=jnp.int32)
    acts=jnp.asarray([c['action'] for c in train],dtype=jnp.int32)
    def sample_real(per_case,offset):
        ii=np.repeat(np.arange(len(train)),per_case)
        y=oracle_batch(jax.random.split(jax.random.PRNGKey(seed+offset),len(ii)),obs[ii],acts[ii])
        return ii,np.asarray(y,dtype=np.int32)
    warm_ids,warm_y=sample_real(32,5000)
    params=init_params(net,meta,case['batch_size'],2,jax.random.PRNGKey(seed))
    warm_opt=optax.adam(3e-3); warm_state=warm_opt.init(params)
    def warm_loss(par,ids,y):
        contexts=jnp.stack([obs[ids],y],axis=1)
        actions=jnp.stack([acts[ids],acts[ids]],axis=1)
        return net.apply(par,contexts,actions,jax.random.PRNGKey(0),method=LatentCNNNetwork.offline_loss)[0]
    @jax.jit
    def warm_step(par,optstate,ids,y):
        loss,grad=jax.value_and_grad(warm_loss)(par,ids,y)
        updates,optstate=warm_opt.update(grad,optstate,par)
        return optax.apply_updates(par,updates),optstate,loss
    r=np.random.default_rng(seed+6000); losses=[]
    for s in range(case['warmup_steps']):
        worker.check_cancelled()
        batch=r.integers(0,len(warm_ids),case['batch_size'])
        params,warm_state,loss=warm_step(params,warm_state,jnp.asarray(warm_ids[batch]),jnp.asarray(warm_y[batch]))
        value=float(loss)
        if not math.isfinite(value):raise ValueError('Non-finite official warmup loss')
        if s%25==0:
            losses.append({'method_id':'warmup','step':s,'loss':value})
            worker.event('progress',{'phase':'official_offline_loss','step':s,'loss':value})
    q0=params
    def logits(par,o,a):return net.apply(par,o,a,method=LatentCNNNetwork.decode)
    @jax.jit
    def log_support(par,o,a):
        lp=jax.nn.log_softmax(logits(par,o,a),axis=-1)
        one=jax.nn.one_hot(jnp.asarray(support),5)
        return jnp.einsum('bct,sct->bs',lp,one)
    q0_train_lp=np.asarray(log_support(q0,obs,acts),dtype=np.float64)
    # Check official comparison implementations directly, without modifying their code.
    official_checks={}
    small_obs=obs[:8];small_acts=acts[:8,None];small_gt=jnp.asarray(warm_y[::32][:8,None,:])
    calls={
      'preferences_loss':lambda par:net.apply(par,small_obs,small_acts,small_acts,small_gt,small_gt,
                        jax.random.PRNGKey(seed+7000),method=LatentCNNNetwork.preferences_loss)[0],
      'k_candidate_preferences_loss':lambda par:net.apply(par,small_obs,small_acts,small_gt,
                        jax.random.PRNGKey(seed+7001),method=LatentCNNNetwork.k_candidate_preferences_loss)[0]}
    for name,fn in calls.items():
        value,grad=jax.jit(jax.value_and_grad(fn))(q0)
        norm=float(optax.global_norm(grad))
        if not math.isfinite(float(value)) or not math.isfinite(norm):raise ValueError('Official loss failed')
        official_checks[name]={'loss':float(value),'gradient_norm':norm}

    ids,real_y=sample_real(12,8000)
    r=np.random.default_rng(seed+9000)
    # Complete independent cell sampling from q0, no conditioning on legal successor set.
    cell_lp=np.asarray(jax.nn.log_softmax(logits(q0,obs,acts),axis=-1),dtype=np.float64)
    cell_p=np.exp(cell_lp);cell_p/=cell_p.sum(-1,keepdims=True)  # full categorical numerical normalization
    candidates=np.empty((len(ids),3,4),dtype=np.int32)
    for gi,ci in enumerate(ids):
        for k in range(3):
            for cell in range(4):candidates[gi,k,cell]=r.choice(5,p=cell_p[ci,cell])
    cand_idx=np.asarray([[index[tuple(y)] for y in group] for group in candidates.tolist()],dtype=np.int32)
    real_idx=np.asarray([index[tuple(y)] for y in real_y.tolist()],dtype=np.int32)
    distances=np.abs(candidates-real_y[:,None,:]).sum(-1)
    winners=np.argmin(distances,axis=1).astype(np.int32)
    strict=distances>np.min(distances,axis=1,keepdims=True)
    label01=np.sign(distances[:,1]-distances[:,0]).astype(np.int32)
    # U=(first winner, strict-loser mask, first-two comparison including ties).
    channel=np.zeros((len(ids),625),dtype=bool)
    groups=[]
    channel_helper_checks=0
    for gi,c in enumerate(candidates):
        all_dist=np.abs(c[None,:,:]-support[:,None,:]).sum(-1)
        ws=np.argmin(all_dist,axis=1)
        masks=all_dist>all_dist.min(-1,keepdims=True)
        labels=np.sign(all_dist[:,1]-all_dist[:,0])
        channel[gi]=(ws==winners[gi]) & (masks==strict[gi]).all(-1) & (labels==label01[gi])
        if not channel[gi,real_idx[gi]]:raise ValueError('Observed feedback absent from its own channel')
        group=make_group(real_y[gi].tolist(),c.tolist(),group_id=f'feedback-{gi:04d}',
            candidate_ids=cand_idx[gi].tolist(),tie_policy='first_best')
        if gi < 3:
            helper_channel=joint_group_channel(support.tolist(),c.tolist(),group,tie_policy='first_best')
            if not np.array_equal(channel[gi],np.asarray(helper_channel,dtype=bool)):
                raise ValueError('Vectorized full-group channel differs from public feedback protocol')
            channel_helper_checks+=1
        group.update(case_id=train[ids[gi]]['case_id'],observed_u={'winner':int(winners[gi]),
            'strict_loser_mask':strict[gi].tolist(),'pair01_label':int(label01[gi])},
            true_query_count=1,distances_role='audit_only_not_likelihood_observation')
        groups.append(group)
    # Fixed minibatches shared by all updates; each optimizer starts freshly from q0.
    schedule=np.random.default_rng(seed+10000).integers(0,len(ids),(case['update_steps'],case['batch_size']))
    train_ids=jnp.asarray(ids);cidx=jnp.asarray(cand_idx);yidx=jnp.asarray(real_idx)
    win=jnp.asarray(winners);mask=jnp.asarray(strict);labels=jnp.asarray(label01);H=jnp.asarray(channel)
    ref=jnp.asarray(q0_train_lp)
    def controlled_loss(par,batch,method):
        ci=train_ids[batch];lp=log_support(par,obs[ci],acts[ci])
        cp=jnp.take_along_axis(lp,cidx[batch],axis=1)
        if method=='mle':return -jnp.mean(jnp.take_along_axis(lp,yidx[batch,None],axis=1))
        if method=='full_group':
            return -jnp.mean(jax.scipy.special.logsumexp(jnp.where(H[batch],lp,-jnp.inf),axis=-1))
        if method in ('bt','dpo'):
            diff=cp[:,0]-cp[:,1]
            if method=='dpo':
                rp=jnp.take_along_axis(ref[ci],cidx[batch],axis=1)
                diff=diff-(rp[:,0]-rp[:,1])
            return jnp.mean(jax.nn.softplus(-labels[batch]*diff)*(labels[batch]!=0))
        best=jnp.take_along_axis(cp,win[batch,None],axis=1)
        per=jax.nn.softplus(-(best-cp))*mask[batch]
        return jnp.mean(per.sum(-1)/jnp.maximum(mask[batch].sum(-1),1))

    # Numerical checks connect vectorized implementations to the public objective helpers.
    check_batch=np.array([int(np.flatnonzero(label01!=0)[0])],dtype=np.int32) if np.any(label01!=0) else np.array([0])
    g=int(check_batch[0]);ci=int(ids[g]);lp=jnp.asarray(q0_train_lp[ci]);c0,c1=map(int,cand_idx[g,:2])
    kernel_checks={}
    if label01[g]:
        wi,li=(c0,c1) if label01[g]>0 else (c1,c0)
        kernel_checks['bt']=abs(float(controlled_loss(q0,jnp.asarray(check_batch),'bt'))-float(objectives.bt_loss(lp,[wi],[li])))
        kernel_checks['dpo']=abs(float(controlled_loss(q0,jnp.asarray(check_batch),'dpo'))-float(objectives.dpo_loss(lp,lp,[wi],[li])))
    kernel_checks['renew_style']=abs(float(controlled_loss(q0,jnp.asarray(check_batch),'renew_style'))-float(objectives.best_of_k_loss(lp,[groups[g]])))
    kernel_checks['mle']=abs(float(controlled_loss(q0,jnp.asarray(check_batch),'mle'))-float(objectives.mle_loss(lp,[int(real_idx[g])])))
    kernel_checks['full_group']=abs(float(controlled_loss(q0,jnp.asarray(check_batch),'full_group'))-float(objectives.group_channel_nll(lp,[channel[g].astype(float).tolist()])))
    if max(kernel_checks.values())>1e-4:raise ValueError('Vectorized losses do not match public objective kernels')

    checkpoints=[];method_rows=[];predictions=[];baseline_decisions={}
    out_ckpt=output/'checkpoints';out_ckpt.mkdir(exist_ok=True)
    test_obs=jnp.asarray([c['board'] for c in test]);test_act=jnp.asarray([c['action'] for c in test])
    for method in case['methods']:
        worker.check_cancelled();par=q0
        if method!='frozen':
            # A single fixed protocol, not a hyperparameter search.
            optimizer=optax.sgd(2e-3);ostate=optimizer.init(par)
            fn=lambda p,b:controlled_loss(p,b,method)
            @jax.jit
            def update(p,os,b):
                v,grad=jax.value_and_grad(fn)(p,b)
                updates,os=optimizer.update(grad,os,p)
                return optax.apply_updates(p,updates),os,v
            for s,batch in enumerate(schedule):
                worker.check_cancelled();par,ostate,value=update(par,ostate,jnp.asarray(batch))
                value=float(value)
                if not math.isfinite(value):raise ValueError('Non-finite controlled update')
                if s%10==0:losses.append({'method_id':method,'step':s,'loss':value})
            worker.event('progress',{'phase':'controlled_update','method':method,'steps':len(schedule),'loss':value})
        path=out_ckpt/(method+'.pkl');save_params(par,str(path))
        restored=load_params(str(path))
        after=np.asarray(log_support(par,test_obs,test_act),dtype=np.float64)
        restored_lp=np.asarray(log_support(restored,test_obs,test_act),dtype=np.float64)
        if not np.array_equal(after,restored_lp):raise ValueError('Checkpoint restore changed predictions')
        checkpoints.append({'path':'checkpoints/'+path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'size_bytes':path.stat().st_size,'method_id':method})
        norm=float(optax.global_norm(jax.tree.map(lambda a,b:a-b,par,q0)))
        rows=[]
        for j,c in enumerate(test):
            # Full cell categorical softmax → complete board probabilities. No candidate renormalization.
            raw_logits=np.asarray(logits(par,test_obs[j:j+1],test_act[j:j+1])[0],dtype=np.float64)
            stable=raw_logits-raw_logits.max(-1,keepdims=True)
            probabilities=np.exp(stable);probabilities/=probabilities.sum(-1,keepdims=True)
            q=np.prod(probabilities[np.arange(4)[None,:],support],axis=1)
            p,event,outcomes=distributions[c['case_id']]
            score=metrics(p.tolist(),q.tolist())
            illegal=float(q[p==0].sum());q_event=float(q[event].sum());p_event=float(p[event].sum())
            decision=decide(q_event,case['risk_threshold'],invalid_mass=illegal,conservative_unknown=True)
            if method=='frozen':baseline_decisions[c['case_id']]=decision['accepted']
            row={'case_id':c['case_id'],'method_id':method,'board':c['board'],'action':c['action'],
                 **score,'invalid_mass':illegal,'true_event_probability':p_event,'predicted_event_probability':q_event,
                 'decision':decision,'decision_changed':decision['accepted']!=baseline_decisions[c['case_id']],
                 'true_accept':p_event<=case['risk_threshold'],
                 'top1_true_probability':float(p[int(np.argmax(q))]),'legal_outcomes':outcomes}
            rows.append(row)
            predictions.append({**row,'p':p.tolist(),'q':q.tolist()})
        avg=lambda key:float(np.mean([r[key] for r in rows]))
        accepted=[r for r in rows if r['decision']['accepted']]
        over=[r for r in rows if not r['true_accept']];under=[r for r in rows if r['true_accept']]
        method_rows.append({'method_id':method,'update_steps':0 if method=='frozen' else case['update_steps'],
            'mean_tv':avg('tv'),'mean_nll':avg('nll'),'mean_brier':avg('expected_categorical_brier'),
            'mean_brier_distribution_error':avg('brier_distribution_error'),'mean_invalid_mass':avg('invalid_mass'),
            'mean_predicted_event_probability':avg('predicted_event_probability'),'mean_true_event_probability':avg('true_event_probability'),
            'acceptance_rate':len(accepted)/len(rows),'decision_changes':sum(r['decision_changed'] for r in rows),
            'parameter_change_norm':norm,'false_acceptance_rate':sum(r['decision']['accepted'] for r in over)/len(over) if over else None,
            'false_rejection_rate':sum(not r['decision']['accepted'] for r in under)/len(under) if under else None,
            'conditional_event_probability':float(np.mean([r['true_event_probability'] for r in accepted])) if accepted else None,
            'information_tier':'realized_successor' if method=='mle' else ('known_channel' if method=='full_group' else 'comparison_or_frozen'),
            'cases':rows})
    archive=output/'checkpoints.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for c in checkpoints:z.write(output/c['path'],c['path'])
    support_file=worker.write_json('support.json',{'schema_version':'feedback_shift.support/1.0','encoding':'base-5 cell exponents','boards':support.tolist()})
    feedback_file=worker.write_json('feedback.json',{'schema_version':'feedback_shift.feedback/1.0','groups':groups,
        'query_count':len(groups),'proposal':'frozen_q0_full_board_distribution','K':3,
        'training_only':True,'audit_truth_is_not_passed_to_comparison_objectives':True})
    prediction_file=worker.write_json('predictions.json',{'schema_version':'feedback_shift.predictions/1.0','support_artifact':'support.json','rows':predictions})
    protocol={'risk_threshold':case['risk_threshold'],'proxy_event':'new_tile_spawns_at_cell_0',
        'event_is_physical_hazard':False,'unknown_policy':'conservative_all_invalid_unsafe',
        'candidate_count':3,'candidate_proposal':'frozen_q0','tie_policy':'first_best_with_strict_loser_mask',
        'observed_feedback':['first_winner','strict_loser_mask','pair01_ternary_label'],
        'judge':'L1 distance in tile exponents','loss_reduction':'mean_of_query_groups_zero_for_all_ties',
        'upstream_k_candidate_reduction':'global_valid_pair_mean; intentionally differs from this controlled group_mean',
        'warmup_optimizer':'Adam','warmup_lr':.003,'update_optimizer':'SGD','update_lr':.002,'beta':1.0,
        'warmup_steps':case['warmup_steps'],'update_steps_per_method':case['update_steps'],'batch_size':case['batch_size'],
        'training_queries':len(ids),'warmup_queries':len(warm_ids),'independent_update_seeds':1,
        'train_contexts':len(train),'evaluation_contexts':len(test),'precision':'float32_training_float64_full_distribution_evaluation',
        'full_group_channel':'exact over all 625 boards for entire defined U; no realized outcome input to this objective',
        'active_uncertainty_querying':False,'selection_aware_composite_executed':False,
        'observed_information_by_method':{'frozen':'none','bt':'pair01 projection','dpo':'pair01 projection and frozen q0',
            'renew_style':'first winner and strict loser mask','mle':'realized successor Y',
            'full_group':'joint U (winner, strict losers, pair01) plus known full channel H'},
        'comparison_limit':'Different feedback projections and privileged-information baselines; not a pure equal-information objective comparison',
        'model_family_limit':'Official per-cell categorical product restricts joint consequence probabilities',
        'brier_definition':'expected categorical Brier over complete support; squared distribution error also saved'}
    inputs=worker.write_json('input.json',{'schema_version':'feedback_shift.input/1.0','case':case,'seed':seed,'train':train,'evaluation':test,'protocol':protocol})
    report={'schema_version':'feedback_shift.report/1.0','scope':'controlled_adaptation',
        'claim_limit':'One-seed 2x2 2048 mechanism pilot with official RENEW network; no full paper reproduction, no ManiSkill safety or language-WM claim.',
        'source':{'commit':config['source_commit'],'path':str(source),
            'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'objective_helpers_sha256':hashlib.sha256(Path(objectives.__file__).read_bytes()).hexdigest(),
            'environment_lock_sha256':config.get('environment_lock_sha256'),
            'network_sha256':hashlib.sha256((source/'world_models/network.py').read_bytes()).hexdigest(),
            'env_sha256':hashlib.sha256((source/'world_models/env.py').read_bytes()).hexdigest()},
        'environment':{'id':'Game2048','board_size':2,'num_tiles':5,'support_size':625,'backend':str(jax.devices()),
            'real_step_support_checks':checked,'no_op_oracle_checks':16},
        'protocol':protocol,'seed':seed,'methods':method_rows,'checkpoints':checkpoints,
        'upstream_loss_checks':official_checks,'objective_kernel_errors':kernel_checks,
        'complete_channel_helper_checks':channel_helper_checks,
        'elapsed_seconds':time.monotonic()-start,'safety_evaluated':False,'losses':losses}
    report_file=worker.write_json('runtime_report.json',report)
    return [{'path':p,'kind':kind,'media_type':media} for p,kind,media in [
        (inputs,'input','application/json'),(support_file,'support','application/json'),
        (feedback_file,'feedback','application/json'),(prediction_file,'predictions','application/json'),
        (report_file,'runtime_report','application/json'),(archive,'checkpoint','application/zip')]]
