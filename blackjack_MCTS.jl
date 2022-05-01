using POMDPs: @gen, isterminal, discount, statetype, actiontype, simulate, states
using POMDPSimulators: RolloutSimulator
using POMDPPolicies: FunctionPolicy
using Statistics: mean, std
using ProgressBars
include("blackjack.jl")

function my_heuristic_policy(m,s)
    # move toward the nearest reward state
    if s[1] > 17 && s[3] == false
        a = :stay
    else
        a = :hit
    end
    return a
end

function my_rollout(m, s)
    r_total = 0.0
    d = 1.0
    while !isterminal(m, s)
        a = my_heuristic_policy(m, s)
        s, r = @gen(:sp,:r)(m, s, a)
        r_total += d*r
        d *= discount(m)
    end
    return r_total
end

bonus(Nsa,Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function my_explore(m,s,N,Q,c)
    Ns = sum(N[(s,a)] for a in m.data.actions)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), m.data.actions)
end


function my_sim!(m,s,N,Q,T,d,c)
    if isterminal(m,s)
        return 0.0
    end
    if d <= 0
        return my_rollout(m, s)
    end
    
    if !haskey(N,(s,first(m.data.actions)))
        for a in m.data.actions
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return my_rollout(m, s)
    end

    a = my_explore(m,s,N,Q,c)
    sprime, r = @gen(:sp,:r)(m, s, a)
    if !haskey(T,(s,a,sprime))
       T[(s,a,sprime)] = 0 
    end
    q = r + discount(m)*my_sim!(m,sprime,N,Q,T,d-1,c)
    N[(s,a)] += 1
    T[(s,a,sprime)] +=1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

# function monteCarloTreeSearch!(m,s,N,Q,T)
#     # Constants
#     n_sims = 7
#     d = 20
#     c = 2.0*(1.0+1.0) # assume a max reward of 1.0 and min reward of -1.0

#     for k in 1:n_sims
#         my_sim!(m,s, N, Q, T, d, c)
#     end
#     return argmax(a->Q[(s,a)],m.data.actions)
# end

# Simulate using heurstic policy
function my_simulate_heuristic(mdp, s0)
    s = s0
    r_total = 0.0
    d = 1.0
    while !isterminal(mdp, s)
        a = my_heuristic_policy(mdp, s)
        s, r = @gen(:sp,:r)(mdp, s, a)
        r_total += d*r
        d *= discount(mdp)
    end
    return r_total
end

# # Select an action using MCTS
# function select_action_mcts(m, s)

#     start = time_ns()
#     n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
#     q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()
#     t = Dict{Tuple{statetype(m), actiontype(m), statetype(m)}, Int}()
#     d = 10
#     c = 2.0*(100-0) # assume a max reward of 100 and min reward of 0
#     while time_ns() < start + 35_000_000 # run for a maximum of 35 ms to leave 15 ms to select an action
#         my_sim!(m,s, n, q, t, d, c)
#     end

#     # select a good action based on q
#     return argmax(a->q[(s,a)],m.data.actions) 
# end

# Select an action using MCTS and log iterations
function select_action_mcts_log_iterations(m, s)

    start = time_ns()
    n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
    q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()
    t = Dict{Tuple{statetype(m), actiontype(m), statetype(m)}, Int}()
    d = 5
    c = 2.0*(1.0+1.0) # assume a max reward of 100 and min reward of 0
    n_iterations = 0
    while time_ns() < start + 20_000_000 # run for a maximum of 40 ms to leave 10 ms to select an action
        my_sim!(m,s, n, q, t, d, c)
        n_iterations += 1
    end

    # select a good action based on q
    if !haskey(q,(s,a))
        return :hit
    else
        return argmax(a->q[(s,a)],m.data.actions), n_iterations
    end
end

# Simulate MCTS
function my_simulate_mcts(mdp, s0)
    s = s0
    r_total = 0.0
    d = 1.0
    n_logged = 0
    n_logged_vec = []
    while !isterminal(mdp, s)
        a, n_logged = select_action_mcts_log_iterations(mdp, s)
        push!(n_logged_vec,n_logged)
        s, r = @gen(:sp,:r)(mdp, s, a)
        r_total += d*r
        d *= discount(mdp)
    end
    return r_total, mean(n_logged_vec)
end
