include("blackjack.jl")
using CommonRLInterface: render, actions, act!, observe, reset!, AbstractEnv, observations, terminated, clone
using CommonRLInterface


function CommonRLInterface.reset!(m)
    m.s = (0,0,false)
end

### double-Q Learning ###

function double_Q_episode!(Q, Q1, Q2, env; eps=0.10, gamma=0.99, alpha=0.1)
    start = time()
    
    function policy(s)
        if rand() < eps
            return rand(actions(env)) # choose random action epsilon fraction of the time
        else
            return argmax(a->Q[(s, a)], actions(env)) # otherwise use greedy policy
        end
    end

    s = env.s
    a = policy(s)
    r = act!(env, a)
    sp = env.s
    hist = Vector{statetype(env.m)}(undef,1)
    hist[1] = s

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
        sp = env.s
        # push!(hist, sp)
        push!(hist, s)

    end

    Q1[(s,a)] += alpha*(r - Q1[(s, a)])
    Q2[(s,a)] += alpha*(r - Q2[(s, a)])

    for s in states(env.m), a in actions(env)
        Q[(s,a)] = (Q1[(s,a)] + Q2[(s,a)])/2.0 # return average of Q1 and Q2
    end

    return (hist=hist, Q = copy(Q1), time=time()-start)
end

function double_Q!(env; n_episodes=100)
    Q0 = 1.0
    Q = Dict((s,a) => Q0 for s in states(env.m), a in actions(env))
    Q1 = Dict((s,a) => Q0 for s in states(env.m), a in actions(env))
    Q2 = Dict((s,a) => Q0 for s in states(env.m), a in actions(env))
    episodes = []
    
    for i in 1:n_episodes
        reset!(env)
        push!(episodes, double_Q_episode!(Q, Q1, Q2, env;
                                          eps= max(0.35,1.0-i/n_episodes))) # use a decaying epsilon
    end
    
    return episodes
end


### SARSA-lambda ###
function sarsa_lambda_episode!(Q, env; ϵ=0.10, γ=1.0, α=0.1, λ=0.95)

    start = time()
    
    function policy(s)
        if rand() < ϵ
            return rand(actions(env)) # choose random action epsilon fraction of the time
        else
            return argmax(a->Q[(s, a)], actions(env)) # otherwise use greedy policy
        end
    end

    s = env.s
    a = policy(s)
    r = act!(env, a)
    sp = env.s
    hist = Vector{statetype(env.m)}(undef,1)
    hist[1] = s
    N = Dict((s, a) => 0.0 for s in states(env.m), a in actions(env))

    while !terminated(env)
        ap = policy(sp)

        N[(s, a)] = get(N, (s, a), 0.0) + 1

        δ = r + γ*Q[(sp, ap)] - Q[(s, a)]

        for ((s, a), n) in N
            Q[(s, a)] += α*δ*n
            N[(s, a)] *= γ*λ
        end

        s = sp
        a = ap
        r = act!(env, a)
        sp = env.s
        push!(hist, sp)
    end

    N[(s, a)] = get(N, (s, a), 0.0) + 1
    δ = r - Q[(s, a)]

    for ((s, a), n) in N
        Q[(s, a)] += α*δ*n
        N[(s, a)] *= γ*λ
    end

    return (hist=hist, Q = copy(Q), time=time()-start)
end

function sarsa_lambda!(env; n_episodes=100, kwargs...)
    Q0 = 1.0
    Q = Dict((s, a) => Q0 for s in states(env.m), a in actions(env))
    episodes = []
    
    for i in 1:n_episodes
        reset!(env)
        push!(episodes, sarsa_lambda_episode!(Q, env;
                                               ϵ=max(0.35, 1.0-i/n_episodes), 
                                            kwargs...))
    end
    
    return episodes
end

### vanilla-Q Learning ###

function vanilla_Q_episode!(Q, env; eps=0.10, gamma=1.0, alpha=0.03)
    start = time()
    
    function policy(s)
        if rand() < eps
            return rand(actions(env)) # choose random action epsilon fraction of the time
        else
            return argmax(a->Q[(s, a)], actions(env)) # otherwise use greedy policy
        end
    end

    s = env.s
    a = policy(s)
    r = act!(env, a)
    sp = env.s
    hist = Vector{statetype(env.m)}(undef,1)
    hist[1] = s

    while !terminated(env)
        ap = policy(sp)
        Q[(s,a)] += alpha*(r + gamma*maximum(a->Q[(sp,a)], actions(env)) - Q[(s,a)])
        s = sp
        a = ap
        r = act!(env, a)
        sp = env.s
        # push!(hist, sp)
        push!(hist, s)

    end

    Q[(s,a)] += alpha*(r - Q[(s, a)])

    return (hist=hist, Q = copy(Q), time=time()-start)
end

function vanilla_Q!(env; n_episodes=100)
    Q0 = 0.0
    Q = Dict((s,a) => Q0 for s in states(env.m), a in actions(env))
    episodes = []
    
    for i in 1:n_episodes
        reset!(env)
        push!(episodes, vanilla_Q_episode!(Q, env;
                                          eps= max(0.35,1.0-i/n_episodes))) # use a decaying epsilon
    end
    
    return episodes
end