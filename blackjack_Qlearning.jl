include("blackjack.jl")
using CommonRLInterface: render, actions, act!, observe, reset!, AbstractEnv, observations, terminated, clone
using CommonRLInterface
using Statistics: mean
using Plots


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
    Q = Dict((s,a) => 0.0 for s in states(env.m), a in actions(env))
    Q1 = Dict((s,a) => 0.0 for s in states(env.m), a in actions(env))
    Q2 = Dict((s,a) => 0.0 for s in states(env.m), a in actions(env))
    episodes = []
    
    for i in 1:n_episodes
        reset!(env)
        push!(episodes, double_Q_episode!(Q, Q1, Q2, env;
                                          eps= max(0.35,0.0))) # use a constant epsilon

    end
    
    return episodes
end


function CommonRLInterface.reset!(m)
    m.s = (5,5,false)
end

# function CommonRLInterface.observe(m::MDPCommonRLEnv)
#     return m.s
# end


env = convert(AbstractEnv, bj)

N = 10000
# N = 50_000
double_Q_episodes = double_Q!(env, n_episodes=N);
# lambda_episodes = sarsa_lambda!(env, n_episodes=N);
finalQ = 
policy = FunctionPolicy(a->actions(bj)[1]) # evaluate always hit policy
policy = FunctionPolicy(my_expert_policy) # evaluate 'expert' policy 
sim = RolloutSimulator(max_steps=100)
@show reward_ = [POMDPs.simulate(sim, bj, policy) for _ in 1:10000]
@show expert_polcy_mean_reward = mean(reward_)




function my_evaluate(env, policy, n_episodes=10000, max_steps=21, γ=1.0)
    returns = Float64[]
    for _ in 1:n_episodes
        t = 0
        r = 0.0
        reset!(env)
        s = env.s
        while !terminated(env)
            a = policy(s)
            r += γ^t*act!(env, a)
            s = env.s
            t += 1
        end
        push!(returns, r)
    end
    return returns
end


plotlyjs()
episodes = Dict("Double-Q"=>double_Q_episodes)
finalQ = episodes["Double-Q"][end].Q
p = plot(xlabel="steps in environment", ylabel="avg return")
n = Int(round(N/20))
stop = N
for (name, eps) in episodes
    Q = Dict((s, a) => 0.0 for s in states(env.m), a in actions(env))
    xs = [0]
    ys = [mean(my_evaluate(env, s->argmax(a->Q[(s, a)], actions(env))))]
    for i in n:n:min(stop, length(eps))
        newsteps = sum(length(ep.hist) for ep in eps[i-n+1:i])
        push!(xs, last(xs) + newsteps)
        Q = eps[i].Q
        push!(ys, mean(my_evaluate(env, s->argmax(a->Q[(s, a)], actions(env)))))
    end    
    plot!(p, xs, ys, label=name, legend=:bottomright, minorticks=true)
end
plot!(p,[xlims(p)[1], xlims(p)[2]], ones(length(xlims(p))).*expert_polcy_mean_reward,label="Expert Policy")
display(p)