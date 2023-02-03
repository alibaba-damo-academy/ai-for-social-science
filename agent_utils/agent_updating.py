def update_agent_profile(env, args, agt_list, agent_name, budget,
                         public_signal_generator,public_signal_to_value):
    # Update the agent info/auction info at the start of the auction
    if args.public_signal:
        # generate public value
        if args.value_to_signal==0: #means from signal to value
            global_signal = public_signal_generator.get_whole_signal_realization()
            public_signal_to_value.next_step(generated_signal_realization=global_signal)

        public_value_gnd = public_signal_to_value.generate_value(obs=None, gnd=True, weighted=False)

        #record
        public_signal_to_value.record_public_gnd_value(public_value_gnd)

        if args.value_to_signal:
            #means from value to signal
            public_signal_generator.generate_signal(data_type='int', upper_bound=public_value_gnd)  # update the signal

            global_signal = public_signal_generator.get_whole_signal_realization()
            public_signal_to_value.next_step(generated_signal_realization=global_signal)

        # each agent observe their public info
        for agt in env.agents:
            agt_list[agt].generate_true_value() # ---> denotes for the private value | may modified with combined public value

    else:
        for agt in env.agents:
            agt_list[agt].generate_true_value()

    # need to update budget

    return
