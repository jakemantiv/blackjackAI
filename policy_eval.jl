include("blackjack_Qlearning.jl")
using CommonRLInterface
using Statistics: mean, std
using Plots
env = convert(AbstractEnv, bj)


function my_evaluate(env, policy, n_episodes=5000, max_steps=21, gamma=1.0)
    returns = Float64[]
    for _ in 1:n_episodes
        t = 0
        r = 0.0
        reset!(env)
        s = env.s
        while !terminated(env)
            a = policy(s)
            r += gamma^t*act!(env, a)
            s = env.s
            t += 1
        end
        push!(returns, r)
    end
    return returns
end

N_expert = 10000
N_doubleQ = 50000
# N = 50_000
println("Starting double Q")
double_Q_episodes = double_Q!(env, n_episodes=N_doubleQ);
println("Starting SARSA lambda")
# sarsa_lambda_Q_episodes = sarsa_lambda!(env, n_episodes=N_doubleQ);
println("Starting Q")
vanilla_Q_episodes = vanilla_Q!(env, n_episodes=N_doubleQ);
println("Starting SARSA")
sarsa_episodes = sarsa!(env, n_episodes=N_doubleQ);


policy = FunctionPolicy(a->actions(bj)[1]) # evaluate always hit policy
policy = FunctionPolicy(my_expert_policy) # evaluate 'expert' policy 
sim = RolloutSimulator(max_steps=100)
reward_ = my_evaluate(env, s->my_expert_policy(s))
expert_polcy_mean_reward = mean(reward_)
expert_policy_standard_error = std(reward_)/sqrt(N_expert)
println("Done with expert")






plotlyjs()
episodes = Dict("Double-Q"=>double_Q_episodes, "SARSA-lambda"=>sarsa_episodes, "Q-learning"=>vanilla_Q_episodes, "SARSA"=>sarsa_episodes)
finalQ = episodes["Double-Q"][end].Q
p = plot(xlabel="steps in environment", ylabel="avg return")
n = Int(round(N_doubleQ/20))
stop = N_doubleQ
N_eval = 5000
for (name, eps) in episodes
    Q = Dict((s, a) => 0.0 for s in states(env.m), a in actions(env))
    xs = [0]
    policy_i = FunctionPolicy(s->argmax(a->Q[(s, a)], actions(env)))
    ys = [mean(my_evaluate(env, s->argmax(a->Q[(s, a)], actions(env))))]
    for i in n:n:min(stop, length(eps))
        newsteps = sum(length(ep.hist) for ep in eps[i-n+1:i])
        push!(xs, last(xs) + newsteps)
        Q = eps[i].Q

        push!(ys, mean(my_evaluate(env, s->argmax(a->Q[(s, a)], actions(env)))))

    end    
    plot!(p, xs, ys, label=name, legend=:bottomright, minorticks=true)
    println("Done with plotting "*name)
end
xlims_ = [xlims(p)[1], xlims(p)[2]]
plot!(p,[xlims_[1], xlims_[2]], ones(length(xlims(p))).*expert_polcy_mean_reward,label="Expert Policy")
plot!(p,[xlims_[1], xlims_[2]], (ones(length(xlims(p))).*expert_polcy_mean_reward).+expert_policy_standard_error*1.96,label="Expert Policy + 95% confidence")
plot!(p,[xlims_[1], xlims_[2]], (ones(length(xlims(p))).*expert_polcy_mean_reward).-expert_policy_standard_error*1.96,label="Expert Policy - 95% confidence")
display(p)