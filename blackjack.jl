using QuickPOMDPs: QuickMDP
using POMDPs
using POMDPModelTools: Deterministic, Uniform, SparseCat
using POMDPPolicies: FunctionPolicy
using POMDPSimulators: RolloutSimulator


# helper functions
cards = [:ace, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] # 4 10s for the face cards + 10
# state[1] = players hand
total_idx = 1
# state[2] = dealer showing
dealer_showing_idx = 2
# state[3] = usable ace?
usable_ace_idx = 3

get_initial_state = function()
    player_draw = rand(cards)
    dealer_draw = rand(cards)
    if player_draw == :ace
        player_total = 11
        usable_ace = true
    else 
        player_total = player_draw
        usable_ace = false
    end
    return (player_total, dealer_draw, usable_ace)
end

bj = QuickMDP(
    actions = [:hit,:stay], # could add double down or surrender
    function(s, a, rng)
        println(s)
        sp_out = (-1, -1, false) # terminal state
        r_out = 0.0
        if s[total_idx] > 21 # went over 21, player automatically loses
            r_out = -1.0
        elseif s[total_idx] == 21 # player hit 21, player automatically wins?
            r_out = 1.0
        elseif a == :stay
            # no change in player count, dealer follows a set strategy
            dealer_turn_over = false
            first_ace = s[dealer_showing_idx] != :ace # was the dealer already showing an ace?
            while !dealer_turn_over
                new_dealer_card = rand(cards)
                if new_dealer_card == :ace
                    first_ace = false
                    if first_ace # first ace counts as 11, subsequent aces count as 1
                        dealer_total += 11
                    else
                        dealer_total += 1  
                    end
                    first_ace = false
                end
                if dealer_total >= 17
                    dealer_turn_over = true
                end
            end

            if dealer_total == s[total_idx]
                r_out = 0.0 # a tie!
            elseif dealer_total > s[total_idx]
                r_out = -1.0 # a loss!
            else # dealer_total < s[total_idx]
                r_out = 1.0 # a win!
            end
            
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
                sp_out = s # no useable aces to save you, will lose next turn
            end
        end
        return (sp=sp_out, r=r_out)
    end,

    # observation = function (s, a, sp)
    #     return s # not a pomdp, the exact state is known
    # end,

    # initialstate = get_initial_state(), # intial draw from the deck
    initialstate = (2,5,false), # intial draw from the deck - test

    discount = 1.0, # not a discounted game
    isterminal = function(s)
        if s[1] == -1 # -1 is our terminal state
            return true
        else 
            return false
        end
    end
)

policy = FunctionPolicy(a->POMDPs.actions(bj)[1]) # evaluate always hit policy
sim = RolloutSimulator(max_steps=100)
@show reward_ = [POMDPs.simulate(sim, bj, policy) for _ in 1:100]
@show mean(reward_)