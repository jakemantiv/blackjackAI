using QuickPOMDPs: QuickMDP
using POMDPs
using POMDPModelTools: Deterministic, Uniform, SparseCat
using POMDPPolicies: FunctionPolicy
using POMDPSimulators: RolloutSimulator


cards = [:ace, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10] # 5 10s for the face cards + 10
# state[1] = players hand 
total_idx = 1
# state[2] = dealer showing 
dealer_showing_idx = 2
# state[3] = usable ace?
usable_ace_idx = 3
bj = QuickMDP(
    actions = [:hit,:stay], # could add double down or surrender
    function (s, a, rng)
        sp_out = [-1, -1, false] # terminal state
        r_out = 0.0
        if s[total_idx] > 21 # went over 21, player automatically loses
            r_out = -1.0
        elseif s[total_idx] == 21 # player hit 21, player automatically wins? 
            r_out = 1.0
        elseif a == :stay 
            # no change in state need to add dealer behavior

            # dealer behavior then reward then terminal state
        elseif a ==:hit 
            new_card = rand(cards)
            if new_card == :ace 
                sp_out[total_idx] = s[total_idx] + 11 # 2 usable aces???
                sp_out[usable_ace_idx] = true
            else
                sp_out[total_idx] = s[total_idx] + new_card
            end
               
        elseif s[total_idx] > 21 # player has gone over 21
            
            if s[usable_ace_idx] == :usable_ace && (s[total_idx] - 11 + 1) <=  21 # if you have a useable ace, use it
                sp_out = s
                sp_out[usable_ace_idx] = false # used up the ace
                sp_out[total_idx] = s[total_idx] - 11 + 1 # ace turns into a 1, update the count
            else 
                sp_out = s # can't do anything about it, will lose next turn
            end
        end
        return (sp=sp_out, r=r_out)
    end,

    # reward = function (s, a)
    #     if s[1] > 21 # player has gone over and loses
    #         return -1.0
    #     elseif s[1] == 21 # player has hit 21 and won
    #         return 1.0
    #     elseif s[1] < 21 # game is still going 
    #         return 0.0
    #     end
    # end,
    observation = function (s, a, sp)
        return s # not a pomdp, the exact state is known
    end,

    # This doesn't work 
    # initialstate = (Uniform([1,2,3,4,5,6,7,8,9,10]), Deterministic(0)), # intial draw from the deck 
    initialstate = [0, 10, false], # intial draw from the deck 
    discount = 1.0, # not a discounted game 
    isterminal = function(s)
        if s[1] == -1 # -1 is our terminal state
            return true
        else 
            return false
        end
    end
)

policy = FunctionPolicy(a->POMDPs.actions(bj)[1]) # always hit policy 
sim = RolloutSimulator(max_steps=100)
@show reward_ = [POMDPs.simulate(sim, bj, policy) for _ in 1:100]
@show mean(reward_)