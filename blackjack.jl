using QuickPOMDPs: QuickMDP
using POMDPs
using POMDPModelTools: Deterministic, Uniform, SparseCat
using POMDPPolicies: FunctionPolicy
using POMDPSimulators: RolloutSimulator



bj = QuickMDP(
    num_card_prob = 1/13,
    ace_card_prob = 1/13,
    face_or_10_prob = 4/13,

    player_idx = 1,
    dealer_idx = 2,

    player_hands = [:5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17, :18, :19, :20, :21,
                :soft13, :soft14, :soft15, :soft16, :soft17, :soft18, :soft19, :soft20, 
                :pair2, :pair3, :pair4, :pair5, :pair6, :pair7, :pair8, :pair9, :pair10, :pairAA,
                :bust],

    dealer_hands = [:A, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17, :18, :19, :20, :21, 
                :soft13, :soft14, :soft15, :soft16, :soft17, :soft18, :soft19, :soft20, 
                :pair2, :pair3, :pair4, :pair5, :pair6, :pair7, :pair8, :pair9, :pair10, :pairAA,
                :bust],

    player_states = [:playing, :staying, :win, :push, :lost],

    states = function (m::QuickMDP)
        s_list = []
        for sp in m.player_states
            for sd in m.dealer_states
                for ps in m.player_states
                    push!(s_list, [(sp, ps), (sd, ps)])
                end
            end 
        end

        return s_list
    end,


    actions = function (s) # could add double down or surrender
        if s[1][2] == :playing   # Can only use :hit while :playing
            return [:hit,:stay]
        elseif s[1][2] == :staying && s[2][2] == :playing
            return [:dealer_hit, :dealer_stay]
        else
            # should only happen when dealer stays
            return [:dealer_stay]
            
        end
            
    end, 

    transition = function (s, a)

        sp = 0  # init to error condition, if-else statements should correctly set

        if s[player_idx][2] == :bust    # player busts, always transition to :lost, transition to terminal states
            sp = Deterministic((s[player_idx][1], :lost), (s[dealer_idx][1], :won))

        elseif s[player_idx][2] == :staying && s[dealer_idx][2] == :busts   # player stays and dealer busts, transition to terminal states
            sp = Deterministic((s[player_idx][1], :won), (s[dealer_idx][1], :lost))
            
        
        elseif s[dealer_idx][2] == :staying && s[player_idx][2] == :staying     # both player and dealer stay... need to resolve higher hand, transition to terminal states
            # TODO need to convert hands to integers
            if s[player_idx][1] > s[dealer_idx][2]
                sp = Deterministic((s[dealer_idx][1], :won),(s[dealer_idx][1], :lost))
            elseif s[player_idx][1] == s[dealer_idx][2]
                sp = Deterministic((s[dealer_idx][1], :push),(s[dealer_idx][1], :push))
            else
                sp = Deterministic((s[dealer_idx][1], :lost),(s[dealer_idx][1], :won))
            end
             
        elseif a == :hit && s[player_idx][2] == :playing        # player hit
            if s[player_idx] == :5
                sp = SparseCat([((:soft16, :playing), s[dealer_idx]), ((:7, :playing), s[dealer_idx]), ((:8, :playing), s[dealer_idx]), ((:9, :playing), s[dealer_idx]), ((:10, :playing), s[dealer_idx]), ((:11, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx])] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :6 
                sp = SparseCat([((:soft17, :playing), s[dealer_idx]), ((:8, :playing), s[dealer_idx]), ((:9, :playing), s[dealer_idx]), ((:10, :playing), s[dealer_idx]), ((:11, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx])] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :7
                sp = SparseCat([((:soft18, :playing), s[dealer_idx]), ((:9, :playing), s[dealer_idx]), ((:10, :playing), s[dealer_idx]), ((:11, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx])] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :8
                sp = SparseCat([((:soft19, :playing), s[dealer_idx]), ((:10, :playing), s[dealer_idx]), ((:11, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx])] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :9
                sp = SparseCat([((:soft20, :playing), s[dealer_idx]), ((:11, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx])] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :10
                sp = SparseCat([((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx])] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :11
                sp = SparseCat([((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :12
                sp = SparseCat([((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :13
                sp = SparseCat([((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+1*num_card_prob)])
            elseif s[player_idx] == :14
                sp = SparseCat([((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+2*num_card_prob)])
            elseif s[player_idx] == :15
                sp = SparseCat([((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+3*num_card_prob)])
            elseif s[player_idx] == :16
                sp = SparseCat([((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+4*num_card_prob)])
            elseif s[player_idx] == :17
                sp = SparseCat([((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+5*num_card_prob)])
            elseif s[player_idx] == :18
                sp = SparseCat([((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+6*num_card_prob)])
            elseif s[player_idx] == :19
                sp = SparseCat([((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,(face_or_10_prob+7*num_card_prob)])
            elseif s[player_idx] == :20
                sp = SparseCat([((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,(face_or_10_prob+8*num_card_prob)])
            elseif s[player_idx] == :21
                sp = Deterministic(((:bust, :lost), s[dealer_idx]))
            elseif s[player_idx] == :soft13
                sp = SparseCat([((:soft14, :playing), s[dealer_idx]), ((:soft15, :playing), s[dealer_idx]), ((:soft16, :playing), s[dealer_idx]), ((:soft17, :playing), s[dealer_idx]), ((:soft18, :playing), s[dealer_idx]), ((:soft19, :playing), s[dealer_idx]), ((:soft20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :soft14
                sp = SparseCat([((:soft15, :playing), s[dealer_idx]), ((:soft16, :playing), s[dealer_idx]), ((:soft17, :playing), s[dealer_idx]), ((:soft18, :playing), s[dealer_idx]), ((:soft19, :playing), s[dealer_idx]), ((:soft20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :soft15
                sp = SparseCat([((:soft16, :playing), s[dealer_idx]), ((:soft17, :playing), s[dealer_idx]), ((:soft18, :playing), s[dealer_idx]), ((:soft19, :playing), s[dealer_idx]), ((:soft20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :soft16
                sp = SparseCat([((:soft17, :playing), s[dealer_idx]), ((:soft18, :playing), s[dealer_idx]), ((:soft19, :playing), s[dealer_idx]), ((:soft20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :soft17
                sp = SparseCat([((:soft18, :playing), s[dealer_idx]), ((:soft19, :playing), s[dealer_idx]), ((:soft20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :soft18
                sp = SparseCat([((:soft19, :playing), s[dealer_idx]), ((:soft20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :soft19
                sp = SparseCat([:soft20, ((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :soft20
                sp = SparseCat([((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :pairAA
                sp = SparseCat([((:soft13, :playing), s[dealer_idx]), ((:soft14, :playing), s[dealer_idx]), ((:soft15, :playing), s[dealer_idx]), ((:soft16, :playing), s[dealer_idx]), ((:soft17, :playing), s[dealer_idx]), ((:soft18, :playing), s[dealer_idx]), ((:soft19, :playing), s[dealer_idx]), ((:soft20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])                
            elseif s[player_idx] == :pair2
                sp = SparseCat([((:soft15, :playing), s[dealer_idx]), :6, ((:7, :playing), s[dealer_idx]), ((:8, :playing), s[dealer_idx]), ((:9, :playing), s[dealer_idx]), ((:10, :playing), s[dealer_idx]), ((:11, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :pair3
                sp = SparseCat([((:soft17, :playing), s[dealer_idx]), ((:8, :playing), s[dealer_idx]), ((:9, :playing), s[dealer_idx]), ((:10, :playing), s[dealer_idx]), ((:11, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :pair4
                sp = SparseCat([((:soft19, :playing), s[dealer_idx]), ((:10, :playing), s[dealer_idx]), ((:11, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :pair5
                sp = SparseCat([((:21, :playing), s[dealer_idx]), ((:12, :playing), s[dealer_idx]), ((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :pair6
                sp = SparseCat([((:13, :playing), s[dealer_idx]), ((:14, :playing), s[dealer_idx]), ((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[player_idx] == :pair7
                sp = SparseCat([((:15, :playing), s[dealer_idx]), ((:16, :playing), s[dealer_idx]), ((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+2*num_card_prob)])
            elseif s[player_idx] == :pair8
                sp = SparseCat([((:17, :playing), s[dealer_idx]), ((:18, :playing), s[dealer_idx]), ((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+4*num_card_prob)])
            elseif s[player_idx] == :pair9
                sp = SparseCat([((:19, :playing), s[dealer_idx]), ((:20, :playing), s[dealer_idx]), ((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+6*num_card_prob)])
            elseif s[player_idx] == :pair10
                sp = SparseCat([((:21, :playing), s[dealer_idx]), ((:bust, :lost), s[dealer_idx])] , [ace_card_prob,(face_or_10_prob+8*num_card_prob)])
            end
                
        elseif a == :stay && s[dealer_idx][2] == :playing   # player stay
            sp = Deterministic((s[player_idx][1], :staying), s[dealer_idx])

        elseif a == :dealer_hit && s[dealer_idx][2] == :playing && s[player_idx][2] == :staying     # player stays, dealer hits
            
            if s[dealer_idx] == :5
                sp = SparseCat([(s[1], (:7, :playing)), (s[1], (:8, :playing)), (s[1], (:9, :playing)), (s[1], (:10, :playing)), (s[1], (:11, :playing)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:soft16, :playing))] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :6 
                sp = SparseCat([(s[1], (:8, :playing)), (s[1], (:9, :playing)), (s[1], (:10, :playing)), (s[1], (:11, :playing)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:soft17, :staying))] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :7
                sp = SparseCat([(s[1], (:9, :playing)), (s[1], (:10, :playing)), (s[1], (:11, :playing)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:soft18, :staying))] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :8
                sp = SparseCat([(s[1], (:10, :playing)), (s[1], (:11, :playing)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:soft19, :staying))] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :9
                sp = SparseCat([(s[1], (:11, :playing)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:soft20, :staying))] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :10
                sp = SparseCat([(s[1], (:21, :staying)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing))] , [ace_card_prob, num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :11
                sp = SparseCat([(s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :12
                sp = SparseCat([(s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :13
                sp = SparseCat([(s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+1*num_card_prob)])
            elseif s[dealer_idx] == :14
                sp = SparseCat([(s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+2*num_card_prob)])
            elseif s[dealer_idx] == :15
                sp = SparseCat([(s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+3*num_card_prob)])
            elseif s[dealer_idx] == :16
                sp = SparseCat([(s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+4*num_card_prob)])
            elseif s[dealer_idx] == :17
                sp = SparseCat([(s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+5*num_card_prob)])
            elseif s[dealer_idx] == :18
                sp = SparseCat([(s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+6*num_card_prob)])
            elseif s[dealer_idx] == :19
                sp = SparseCat([(s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,(face_or_10_prob+7*num_card_prob)])
            elseif s[dealer_idx] == :20
                sp = SparseCat([(s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,(face_or_10_prob+8*num_card_prob)])
            elseif s[dealer_idx] == :soft13
                sp = SparseCat([(s[1], (:soft14, :playing)), (s[1], (:soft15, :playing)), (s[1], (:soft16, :playing)), (s[1], (:soft17, :staying)), (s[1], (:soft18, :staying)), (s[1], (:soft19, :staying)), (s[1], (:soft20, :staying)), (s[1], (:21, :staying)), (s[1], (:12, :playing)), (s[1], (:13, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :soft14
                sp = SparseCat([(s[1], (:soft15, :playing)), (s[1], (:soft16, :playing)), (s[1], (:soft17, :staying)), (s[1], (:soft18, :staying)), (s[1], (:soft19, :staying)), (s[1], (:soft20, :staying)), (s[1], (:21, :staying)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :soft15
                sp = SparseCat([(s[1], (:soft16, :playing)), (s[1], (:soft17, :staying)), (s[1], (:soft18, :staying)), (s[1], (:soft19, :staying)), (s[1], (:soft20, :staying)), (s[1], (:21, :staying)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :soft16
                sp = SparseCat([(s[1], (:soft17, :staying)), (s[1], (:soft18, :staying)), (s[1], (:soft19, :staying)), (s[1], (:soft20, :staying)), (s[1], (:21, :staying)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :soft17
                sp = SparseCat([(s[1], (:soft18, :staying)), (s[1], (:soft19, :staying)), (s[1], (:soft20, :staying)), (s[1], (:21, :staying)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :soft18
                sp = SparseCat([(s[1], (:soft19, :staying)), (s[1], (:soft20, :staying)), (s[1], (:21, :staying)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :soft19
                sp = SparseCat([:soft20, (s[1], (:21, :staying)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :soft20
                sp = SparseCat([(s[1], (:21, :staying)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :pairAA
                sp = SparseCat([(s[1], (:soft13, :playing)), (s[1], (:soft14, :playing)), (s[1], (:soft15, :playing)), (s[1], (:soft16, :playing)), (s[1], (:soft17, :staying)), (s[1], (:soft18, :staying)), (s[1], (:soft19, :staying)), (s[1], (:soft20, :staying)), (s[1], (:21, :staying)), (s[1], (:12, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])                
            elseif s[dealer_idx] == :pair2
                sp = SparseCat([(s[1], (:soft15, :playing)), :6, (s[1], (:7, :playing)), (s[1], (:8, :playing)), (s[1], (:9, :playing)), (s[1], (:10, :playing)), (s[1], (:11, :playing)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :pair3
                sp = SparseCat([(s[1], (:soft17, :staying)), (s[1], (:8, :playing)), (s[1], (:9, :playing)), (s[1], (:10, :playing)), (s[1], (:11, :playing)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :pair4
                sp = SparseCat([(s[1], (:soft19, :staying)), (s[1], (:10, :playing)), (s[1], (:11, :playing)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :pair5
                sp = SparseCat([(s[1], (:21, :staying)), (s[1], (:12, :playing)), (s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :pair6
                sp = SparseCat([(s[1], (:13, :playing)), (s[1], (:14, :playing)), (s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,face_or_10_prob])
            elseif s[dealer_idx] == :pair7
                sp = SparseCat([(s[1], (:15, :playing)), (s[1], (:16, :playing)), (s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+2*num_card_prob)])
            elseif s[dealer_idx] == :pair8
                sp = SparseCat([(s[1], (:17, :playing)), (s[1], (:18, :playing)), (s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+4*num_card_prob)])
            elseif s[dealer_idx] == :pair9
                sp = SparseCat([(s[1], (:19, :playing)), (s[1], (:20, :playing)), (s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,num_card_prob,num_card_prob,(face_or_10_prob+6*num_card_prob)])
            elseif s[dealer_idx] == :pair10
                sp = SparseCat([(s[1], (:21, :staying)), (s[1], (:bust, :lost))] , [ace_card_prob,(face_or_10_prob+8*num_card_prob)])
            end

        else
            println("Error")
            sp = 0 # error condition
        end

        return sp
    end,

    reward = function (s, a)
        if s[player_idx][2] == :lost    # :lost is terminal state
            return -1.0
        elseif s[player_idx][2] == :won # :won is terminal state
            return 1.0
        else
            return 0.0
        end
    end,
    
    observation = function (s, a, sp)
        return s # not a pomdp, the exact state is known
    end,

    initialstate = Uniform(states), # intial draw from the deck, TODO: correct for drawing 21
    discount = 1.0, # not a discounted game 

    isterminal = function (s)
        if s[player_idx][2] == :won || s[player_idx][2] == :lost || s[player_idx][2] == :push
            return true
        else 
            return false
        end
    end,
)

policy = FunctionPolicy(a->POMDPs.actions(bj)[1]) # always hit policy 
sim = RolloutSimulator(max_steps=100)
@show reward_ = [POMDPs.simulate(sim, bj, policy) for _ in 1:100]
@show mean(reward_)