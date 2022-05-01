include("blackjack_Qlearning.jl")
include("blackjack_MCTS.jl")
using CommonRLInterface
using Statistics: mean, std
using Plots
env = convert(AbstractEnv, bj)


function my_evaluate(env, policy, n_episodes=10000, max_steps=21, gamma=1.0)
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
GC.gc() # garbage cleanup
N_expert = 100000
N_doubleQ = 100000
# N = 50_000
println("Starting double Q")
double_Q_episodes = double_Q!(env, n_episodes=N_doubleQ);
println("Starting SARSA lambda")
sarsa_lambda_Q_episodes = sarsa_lambda!(env, n_episodes=N_doubleQ);
# sarsa_lambda_Q_episodes = double_Q_episodes;

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

episodes = Dict("Double-Q"=>double_Q_episodes, "SARSA-lambda"=>sarsa_episodes, "Q-learning"=>vanilla_Q_episodes, "SARSA"=>sarsa_episodes)
finalQ = episodes["Double-Q"][end].Q
plotlyjs()
p = plot(xlabel="steps in environment", ylabel="avg return")
n = Int(round(N_doubleQ/20))
stop = N_doubleQ
N_eval = 10000
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

q = plot(xlabel="wall clock time (s)", ylabel="avg return")
n = Int(round(N_doubleQ/20))
stop = N_doubleQ
for (name,eps) in episodes
    Q = Dict((s, a) => 0.0 for s in states(env.m), a in actions(env))
    xs = [0.0]
    ys = [mean(my_evaluate(env, s->argmax(a->Q[(s, a)], actions(env))))]
    for i in n:n:min(stop, length(eps))
        newtime = sum(ep.time for ep in eps[i-n+1:i])
        push!(xs, last(xs) + newtime)
        Q = eps[i].Q
        push!(ys, mean(my_evaluate(env, s->argmax(a->Q[(s, a)], actions(env)))))
    end    
    plot!(q, xs, ys, label=name)
    println("Done with plotting "*name)
end
xlims_ = [xlims(q)[1], xlims(q)[2]]
plot!(q,[xlims_[1], xlims_[2]], ones(length(xlims(p))).*expert_polcy_mean_reward,label="Expert Policy")
plot!(q,[xlims_[1], xlims_[2]], (ones(length(xlims(p))).*expert_polcy_mean_reward).+expert_policy_standard_error*1.96,label="Expert Policy + 95% confidence")
plot!(q,[xlims_[1], xlims_[2]], (ones(length(xlims(p))).*expert_polcy_mean_reward).-expert_policy_standard_error*1.96,label="Expert Policy - 95% confidence")
display(q)

### Tabulate Rewards ###
println("Expert Mean Reward = " * string(expert_polcy_mean_reward))
println("Expert Standard error of the mean reward = "*string(expert_policy_standard_error))
println("Expert 95% confidence = " * string(1.96*expert_policy_standard_error))
println("Expert Win Rate = "* string(count(i->(i==1.0),reward_)/length(reward_)))
println("")
N_heuristic = 10000
N_MCTS = 10000
s0 = (0,0,false)
# heuristic policy eval
R_heuristic = Vector{Float64}()
for k in 1:N_heuristic
    r = my_simulate_heuristic(bj,s0)
    push!(R_heuristic,r)
end
println("Heuristic Mean Reward = " * string(mean(R_heuristic)))
println("Heuristic Standard error of the mean reward = " * string(std(R_heuristic)/sqrt(N_heuristic)))
println("Heuristic 95% confidence = " * string(1.96*std(R_heuristic)/sqrt(N_heuristic)))
println("Heuristic Win Rate = "* string(count(i->(i==1.0),R_heuristic)/length(R_heuristic)))

println("")
# mcts policy eval
R_mcts = Vector{Float64}()
for k in ProgressBar(1:N_MCTS)
    r, avg_iterations = my_simulate_mcts(bj,s0)
    push!(R_mcts,r)
end
println("MCTS Mean Reward = " * string(mean(R_mcts)))
println("MCTS Standard error of the mean reward = " * string(std(R_mcts)/sqrt(N_MCTS)))
println("MCTS 95% confidence = " * string(1.96*std(R_mcts)/sqrt(N_MCTS)))
println("MCTS Win Rate = "* string(count(i->(i==1.0),R_mcts)/length(R_mcts)))

println("")


for (name,eps) in episodes
    Q = eps[end].Q
    reward_eval = my_evaluate(env, s->argmax(a->Q[(s, a)], actions(env)))
    println(name *" Mean Reward = " * string(mean(reward_eval)))
    println(name *" Standard error of the mean reward = " * string(std(reward_eval)/sqrt(N_eval)))
    println(name *" 95% confidence = " * string(1.96*std(reward_eval)/sqrt(N_eval)))
    println(name *" Win Rate = "* string(count(i->(i==1.0),reward_eval)/length(reward_eval)))
    println("")
end