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
# state[3] = useable ace?
useable_ace_idx = 3

get_initial_state = function()
    player_draw = rand(cards)
    dealer_draw = rand(cards)
    if player_draw == :ace
        player_total = 11
        useable_ace = true
    else 
        player_total = player_draw
        useable_ace = false
    end
    return (player_total, dealer_draw, useable_ace)
end

function my_expert_policy(s)
    player_total_in = s[total_idx]
    dealer_showing_in = s[dealer_showing_idx]
    useable_ace_in = s[useable_ace_idx]

    if useable_ace_in == false 
        if player_total_in <= 11
            action_out = :hit
        elseif player_total_in == 12
            if dealer_showing_in in [2,3,7,8,9,10,:ace]
                action_out = :hit
            else 
                action_out = :stay
            end
        elseif player_total_in in [13, 14, 15, 16]
            if dealer_showing_in in [7,8,9,10,:ace]
                action_out = :hit
            else 
                action_out = :stay
            end
        elseif player_total_in >= 17
            action_out = :stay
        end
    else #useable_ace_in == true
        if player_total_in <= 17
            action_out = :hit
        elseif player_total_in == 18
            if dealer_showing_in in [9,10,:ace]
                action_out = :hit
            else 
                action_out = :stay
    
            end
        elseif player_total_in >= 19
            action_out = :stay
        end
    end
    return action_out
end

bj = QuickMDP(
    actions = [:hit,:stay], # could add double down or surrender
    function(s, a, rng)
        player_total_in = s[total_idx]
        dealer_showing_in = s[dealer_showing_idx]
        useable_ace_in = s[useable_ace_idx]

        player_total_out = -1
        dealer_showing_out = -1
        useable_ace_out = false

        r_out = 0.0
        if s[total_idx] > 21 # went over 21, player automatically loses
            r_out = -1.0
            sp_out = (player_total_out, dealer_showing_out, useable_ace_out)
        elseif s[total_idx] == 21 # player hit 21, player automatically wins?
            r_out = 1.0
            sp_out = (player_total_out, dealer_showing_out, useable_ace_out)
        elseif a == :stay
            # no change in player count, dealer follows a set strategy
            dealer_turn_over = false
            first_ace = dealer_showing_in != :ace # was the dealer already showing an ace?

            if dealer_showing_in == :ace
                dealer_total = 11
            else
                dealer_total = dealer_showing_in
            end

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
                else
                    dealer_total += new_dealer_card 
                end
                if dealer_total >= 17
                    dealer_turn_over = true
                end
            end

            if dealer_total == player_total_in
                r_out = 0.0 # a tie!
            elseif dealer_total > 21 
                r_out = 1.0 # dealer busts, a win!
            elseif dealer_total > player_total_in
                r_out = -1.0 # a loss!
            else # dealer_total < s[total_idx]
                r_out = 1.0 # a win!
            end
            sp_out = (player_total_out, dealer_showing_in, useable_ace_out)

        elseif a ==:hit 
            new_card = rand(cards)
            if new_card == :ace 
                player_total_out = player_total_in + 11  # 2 useable aces???
                useable_ace_out = true
                dealer_showing_out = dealer_showing_in
            else
                player_total_out = player_total_in + new_card
                useable_ace_out = useable_ace_in
                dealer_showing_out = dealer_showing_in
            end
            sp_out = (player_total_out, dealer_showing_out, useable_ace_out)

        elseif s[total_idx] > 21 # player has gone over 21
            
            if s[useable_ace_idx] == true && (s[total_idx] - 11 + 1) <=  21 # if you have a useable ace, use it
                player_total_out = player_total_in - 11 + 1 # ace turns into a 1, update the count
                dealer_showing_out = dealer_showing_in
                useable_ace_out = false # used up the ace
            else 
            end
            sp_out = (player_total_out, dealer_showing_out, useable_ace_out)

        end
        return (sp=sp_out, r=r_out)
    end,

    # initialstate = get_initial_state(), # intial draw from the deck
    initialstate = Deterministic((5,5,false)), # intial draw from the deck - test
    # initialstate = Uniform(([2,3,4,5,6,7,8,9,10], [2,3,4,5,], [true,false])),
    discount = 1.0, # not a discounted game
    isterminal = function(s)
        if s[1] == -1 # -1 is our terminal state
            return true
        else 
            return false
        end
    end
)
# policy = FunctionPolicy(a->actions(bj)[1]) # evaluate always hit policy
policy = FunctionPolicy(my_expert_policy) # evaluate 'expert' policy 
sim = RolloutSimulator(max_steps=100)
@show reward_ = [POMDPs.simulate(sim, bj, policy) for _ in 1:1000]
@show mean(reward_)