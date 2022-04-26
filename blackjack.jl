using QuickPOMDPs: QuickMDP
using POMDPs
using POMDPModelTools: Deterministic, Uniform, SparseCat, ImplicitDistribution

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
    #  https://wizardofodds.com/games/blackjack/strategy/4-decks/
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

function simulate_dealer(dealer_showing_in)
    dealer_turn_over = false
    dealer_useable_ace = dealer_showing_in == :ace # was the dealer already showing an ace?
    dealer_blackjack = false
    num_turns = 1

    if dealer_showing_in == :ace
        dealer_total = 11
    else
        dealer_total = dealer_showing_in
    end

    while !dealer_turn_over
        num_turns +=1 # keep count of draws for natural blackjack
        new_dealer_card = rand(cards) # draw a new card
        if new_dealer_card == :ace 
            if num_turns == 2 && (dealer_total + 11) == 21
                dealer_blackjack = true 
                dealer_total += 11 # natural blackjack
            elseif dealer_useable_ace == true && (dealer_total + 11) > 21
                dealer_total += 1 + 1 -11
                dealer_useable_ace = false 
            elseif dealer_useable_ace == true && (dealer_total + 11) <= 21
                dealer_total += 11
                dealer_useable_ace = true 
            elseif dealer_useable_ace == false && (dealer_total + 11) <= 21
                dealer_total += 11
                dealer_useable_ace = true 
            elseif dealer_useable_ace == false && (dealer_total + 11) > 21 
                dealer_total += 1
            else
                println(dealer_showing_in) 
            end

        else
            if num_turns == 2 && (dealer_total + new_dealer_card) == 21
                dealer_blackjack = true 
                dealer_total += new_dealer_card # natural blackjack
            elseif dealer_useable_ace == true && (dealer_total + new_dealer_card) > 21
                dealer_total += new_dealer_card - 11 + 1
                dealer_useable_ace = false 
            else
                dealer_total += new_dealer_card 
            end
        end
        if dealer_total >= 17 || dealer_blackjack # stand on soft 17
            dealer_turn_over = true
        end
    end

    return (dealer_blackjack, dealer_total)
end

bj = QuickMDP(
    actions = [:hit,:stay], # could add double down or surrender or split 
    states = [(p_count,d_show,use_ace) for p_count in -1:31 for d_show in cards for use_ace in [true, false]],
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
        elseif s[total_idx] == 21 # player hit 21, player automatically wins unless dealer blackjacks? 
            dealer_blackjack, dealer_total = simulate_dealer(dealer_showing_in)
            if dealer_blackjack
                r_out = 0.0
            else
                r_out = 1.0
            end
            sp_out = (player_total_out, dealer_showing_out, useable_ace_out)
        elseif a == :stay
            # no change in player count, dealer follows a set strategy
            dealer_blackjack, dealer_total = simulate_dealer(dealer_showing_in)

            if dealer_total == player_total_in
                r_out = 0.0 # a tie!
            elseif dealer_total > 21 
                r_out = 1.0 # dealer busts, a win!
            elseif dealer_total > player_total_in
                r_out = -1.0 # a loss!
            else # dealer_total < s[total_idx]
                r_out = 1.0 # a win!
            end
            sp_out = (player_total_out, dealer_showing_out, useable_ace_out)

        elseif a ==:hit 
            new_card = rand(cards)

            if useable_ace_in == true 
                if new_card == :ace && ((player_total_in + 1) <= 21)
                    player_total_out = player_total_in + 1
                    dealer_showing_out = dealer_showing_in
                    useable_ace_out = true
                    # accept the ace as an 1 value, still maintain the useable ace
                elseif new_card == :ace && ((player_total_in + 1) > 21)
                    player_total_out = player_total_in - 11 + 1
                    dealer_showing_out = dealer_showing_in
                    useable_ace_out = false
                    # accept the ace as an 1 value, use the useable ace
                elseif new_card != :ace && ((player_total_in + new_card) > 21)
                    player_total_out = player_total_in - 11 + 1 + new_card
                    dealer_showing_out = dealer_showing_in
                    useable_ace_out = false
                else # new_card != :ace
                    player_total_out = player_total_in + new_card
                    dealer_showing_out = dealer_showing_in
                    useable_ace_out = useable_ace_in
                end
            else
                if new_card == :ace && ((player_total_in + 11) <= 21) 
                    player_total_out = player_total_in + 11
                    dealer_showing_out = dealer_showing_in
                    useable_ace_out = true
                    # accept the ace as an 11 value, now have a useable ace
                elseif new_card == :ace && ((player_total_in + 11) > 21)
                    player_total_out = player_total_in + 1
                    dealer_showing_out = dealer_showing_in
                    useable_ace_out = false
                    # accepting the ace as an 11 would put us over, use the ace as a 1, still no usable ace 
                else #new_card != :ace
                    player_total_out = player_total_in + new_card
                    dealer_showing_out = dealer_showing_in
                    useable_ace_out = false
                end
            end
            sp_out = (player_total_out, dealer_showing_out, useable_ace_out)


        end
        return (sp=sp_out, r=r_out)
    end,


    observation = function (s)
        return Deterministic(s) # not a pomdp, the exact state is known

    end,
    # initialstate = get_initial_state(), # intial draw from the deck
    initialstate = Deterministic((5,5,false)), # intial draw from the deck - test
    # initialstate = ImplicitDistribution() do rng
    #             player_card = rand(cards)
    #             dealer_card = rand(cards)
    #             if player_card == :ace
    #                 player_total = 11
    #             else
    #                 player_total = player_card
    #             end
    #             return (player_total, dealer_card, player_card==:ace)
    #         end,

    discount = 1.0, # not a discounted game
    isterminal = function(s)
        if s[1] == -1 # -1 is our terminal state
            return true
        else
            return false
        end
    end,
)
