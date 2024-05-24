import samna.dynapse1 as dyn1

def agent_param_group():
    """Generate a Dynapse1ParameterGroup of one core with some synapse
    weights turned on for examples.

    Returns:
        samna.dynapse1.Dynapse1ParameterGroup: Dynapse1ParameterGroup.
    """
    param_group = dyn1.Dynapse1ParameterGroup()
    # Neuron V_mem gain (not the firing threshold!) #chip 07
    param_group.param_map["IF_THR_N"].coarse_value = 0 # lower than 2-3
    param_group.param_map["IF_THR_N"].fine_value = 0
    # # refactory period
    param_group.param_map["IF_RFR_N"].coarse_value = 2 # the lower the bigger the refactory period
    param_group.param_map["IF_RFR_N"].fine_value = 128
    # # Main neuron time constant (leakage current)
    param_group.param_map["IF_TAU1_N"].coarse_value = 3 # higher than 2-3
    param_group.param_map["IF_TAU1_N"].fine_value = 150
    # # Secondary neuron time constant
    param_group.param_map["IF_TAU2_N"].coarse_value = 3
    param_group.param_map["IF_TAU2_N"].fine_value = 150
    # # DC current injected into a neuron
    param_group.param_map["IF_DC_P"].coarse_value = 0
    param_group.param_map["IF_DC_P"].fine_value = 0
    
    ### AMPA snyapse ###
    # time constant (leakage)
    param_group.param_map["NPDPIE_TAU_F_P"].coarse_value = 7
    param_group.param_map["NPDPIE_TAU_F_P"].fine_value = 255
    # synapses threshold (gain)
    param_group.param_map["NPDPIE_THR_F_P"].coarse_value = 5
    param_group.param_map["NPDPIE_THR_F_P"].fine_value = 200
    # synapse weight
    param_group.param_map["PS_WEIGHT_EXC_F_N"].coarse_value = 6
    param_group.param_map["PS_WEIGHT_EXC_F_N"].fine_value = 200
    
    ### NMDA snyapse ###
    # time constant (leakage)
    param_group.param_map["NPDPIE_TAU_S_P"].coarse_value = 2
    param_group.param_map["NPDPIE_TAU_S_P"].fine_value = 100
    # # synapses threshold
    param_group.param_map["NPDPIE_THR_S_P"].coarse_value = 5
    param_group.param_map["NPDPIE_THR_S_P"].fine_value = 200
    # # synapses weight
    param_group.param_map["PS_WEIGHT_EXC_S_N"].coarse_value = 5
    param_group.param_map["PS_WEIGHT_EXC_S_N"].fine_value = 47

    ### GABA_A snyapse ###
    # time constant (leakage)
    param_group.param_map["NPDPII_TAU_F_P"].coarse_value = 7
    param_group.param_map["NPDPII_TAU_F_P"].fine_value = 255
    # # synapses threshold
    param_group.param_map["NPDPII_THR_F_P"].coarse_value = 5
    param_group.param_map["NPDPII_THR_F_P"].fine_value = 200
    # # synapse weights
    param_group.param_map["PS_WEIGHT_INH_F_N"].coarse_value = 6
    param_group.param_map["PS_WEIGHT_INH_F_N"].fine_value = 120

    
    # ### GABA_B snyapse ###
    # # time constant (leakage)
    param_group.param_map["NPDPII_TAU_S_P"].coarse_value = 7
    param_group.param_map["NPDPII_TAU_S_P"].fine_value = 255
    # # synapses threshold
    param_group.param_map["NPDPII_THR_S_P"].coarse_value = 5
    param_group.param_map["NPDPII_THR_S_P"].fine_value = 200
    # # synapse weights
    param_group.param_map["PS_WEIGHT_INH_S_N"].coarse_value = 7
    param_group.param_map["PS_WEIGHT_INH_S_N"].fine_value = 255

    return param_group

def model_param_group():
    """Generate a Dynapse1ParameterGroup of one core with some synapse
    weights turned on for examples.

    Returns:
        samna.dynapse1.Dynapse1ParameterGroup: Dynapse1ParameterGroup.
    """
    param_group = dyn1.Dynapse1ParameterGroup()
    # Neuron V_mem gain (not the firing threshold!) 
    param_group.param_map["IF_THR_N"].coarse_value = 0 # lower than 2-3
    param_group.param_map["IF_THR_N"].fine_value = 0
    # # refactory period
    param_group.param_map["IF_RFR_N"].coarse_value = 2 # the lower the bigger the refactory period
    param_group.param_map["IF_RFR_N"].fine_value = 128
    # # Main neuron time constant (leakage current)
    param_group.param_map["IF_TAU1_N"].coarse_value = 3 # higher than 2-3
    param_group.param_map["IF_TAU1_N"].fine_value = 150
    # # Secondary neuron time constant
    param_group.param_map["IF_TAU2_N"].coarse_value = 3
    param_group.param_map["IF_TAU2_N"].fine_value = 150
    # # DC current injected into a neuron
    param_group.param_map["IF_DC_P"].coarse_value = 0
    param_group.param_map["IF_DC_P"].fine_value = 0
    
    ### AMPA snyapse ###
    # time constant (leakage)
    param_group.param_map["NPDPIE_TAU_F_P"].coarse_value = 7
    param_group.param_map["NPDPIE_TAU_F_P"].fine_value = 255
    # synapses threshold (gain)
    param_group.param_map["NPDPIE_THR_F_P"].coarse_value = 5
    param_group.param_map["NPDPIE_THR_F_P"].fine_value = 200
    # synapse weight
    param_group.param_map["PS_WEIGHT_EXC_F_N"].coarse_value = 6
    param_group.param_map["PS_WEIGHT_EXC_F_N"].fine_value = 200
    
    ### NMDA snyapse ###
    # time constant (leakage)
    param_group.param_map["NPDPIE_TAU_S_P"].coarse_value = 2
    param_group.param_map["NPDPIE_TAU_S_P"].fine_value = 100
    # # synapses threshold
    param_group.param_map["NPDPIE_THR_S_P"].coarse_value = 5
    param_group.param_map["NPDPIE_THR_S_P"].fine_value = 200
    # # synapses weight
    param_group.param_map["PS_WEIGHT_EXC_S_N"].coarse_value = 4
    param_group.param_map["PS_WEIGHT_EXC_S_N"].fine_value = 50

    ### GABA_A snyapse ###
    # time constant (leakage)
    param_group.param_map["NPDPII_TAU_F_P"].coarse_value = 3
    param_group.param_map["NPDPII_TAU_F_P"].fine_value = 150
    # # synapses threshold
    param_group.param_map["NPDPII_THR_F_P"].coarse_value = 5
    param_group.param_map["NPDPII_THR_F_P"].fine_value = 200
    # # synapse weights
    param_group.param_map["PS_WEIGHT_INH_F_N"].coarse_value = 6
    param_group.param_map["PS_WEIGHT_INH_F_N"].fine_value = 120

    # ### GABA_B snyapse ###
    # # time constant (leakage)
    param_group.param_map["NPDPII_TAU_S_P"].coarse_value = 3
    param_group.param_map["NPDPII_TAU_S_P"].fine_value = 150
    # # synapses threshold
    param_group.param_map["NPDPII_THR_S_P"].coarse_value = 5
    param_group.param_map["NPDPII_THR_S_P"].fine_value = 200
    # # synapse weights
    param_group.param_map["PS_WEIGHT_INH_S_N"].coarse_value = 7
    param_group.param_map["PS_WEIGHT_INH_S_N"].fine_value = 255


    return param_group