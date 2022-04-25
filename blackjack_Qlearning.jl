include("blackjack.jl")
using CommonRLInterface: render, actions, act!, observe, reset!, AbstractEnv, observations, terminated, clone



function double_Q_episode!(Q, Q1, Q2, env; eps=0.10, gamma=0.99, alpha=0.1)
    start = time()
    
    function policy(s)
        if rand() < eps
            return rand(actions(env)) # choose random action epsilon fraction of the time
        else
            return argmax(a->Q[(s, a)], actions(env)) # otherwise use greedy policy
        end
    end

    s = observe(env)
    a = policy(s)
    r = act!(env, a)
    sp = observe(env)
    hist = [s]

    while !terminated(env)
        ap = policy(sp)
        if rand() < 0.5
            # update Q1 half of the time
            Q1[(s,a)] += alpha*(r + gamma*Q2[sp, argmax(a->Q1[sp,a], actions(env))] - Q1[(s, a)])
        else
            # update Q2 the other half
            Q2[(s,a)] += alpha*(r + gamma*Q1[sp, argmax(a->Q2[sp,a], actions(env))] - Q2[(s, a)])
        end
        s = sp
        a = ap
        r = act!(env, a)
        sp = observe(env)
        push!(hist, sp)
    end

    Q1[(s,a)] += alpha*(r - Q1[(s, a)])
    Q2[(s,a)] += alpha*(r - Q2[(s, a)])

    for s in observations(env), a in actions(env)
        Q[(s,a)] = (Q1[(s,a)] + Q2[(s,a)])/2.0 # return average of Q1 and Q2
    end

    return (hist=hist, Q = copy(Q), time=time()-start)
end

function double_Q!(env; n_episodes=100)
    Q = Dict((s,a) => 0.0 for s in observations(env), a in actions(env))
    Q1 = Dict((s,a) => 0.0 for s in observations(env), a in actions(env))
    Q2 = Dict((s,a) => 0.0 for s in observations(env), a in actions(env))
    episodes = []
    
    for i in 1:n_episodes
        reset!(env)
        push!(episodes, double_Q_episode!(Q, Q1, Q2, env;
                                          eps= max(0.35,0.0))) # use a constant epsilon

    end
    
    return episodes
end

# env = convert(AbstractEnv, bj)

# N = 500_000
# # N = 500_000
# double_Q_episodes = double_Q!(env, n_episodes=N);
# lambda_episodes = sarsa_lambda!(env, n_episodes=N);

policy = FunctionPolicy(a->actions(bj)[1]) # evaluate always hit policy
policy = FunctionPolicy(my_expert_policy) # evaluate 'expert' policy 
sim = RolloutSimulator(max_steps=100)
@show reward_ = [POMDPs.simulate(sim, bj, policy) for _ in 1:10000]
@show mean(reward_)