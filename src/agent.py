import samna.dynapse1 as dyn1
import dynapse.dynapse1utils as ut
import dynapse.netgen as n
import src.params as params

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.optimizer import Adam

import time 
import random
import configparser

class PongAgent:
    def __init__(self, model, if_dream):
        # Read config
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.num_actions = self.config.getint('AGENT', 'num_actions')
        self.num_hidden_neurons = self.config.getint('AGENT', 'num_hidden_neurons')
        self.spike_gens_per_input_value = self.config.getint('AGENT', 'spike_gens_per_input_value')
        self.if_dream = self.config.getboolean('AGENT', 'if_dream')
        self.if_dream = if_dream

        # Initialize readout
        self.init_policy_readout()
        if self.if_dream:
            self.init_model_readout()
        self.latest_spikes = []
        self.latest_spike_rates = []

        # Store DYNAP-SE1 object
        self.model = model

        # Set target chip for spike generators
        self.target_chip = self.config.getint('DEFAULT', 'chip_id')

        self.num_spikegens = 4*self.spike_gens_per_input_value
        self.spikegen_ids = range(self.num_spikegens + 3)
        
        # Define network configuration
        self.net_gen = n.NetworkGenerator()
        self.monitored_neurons = []
        self.init_policy_net()
        if self.if_dream:
            self.init_model_net()

        # Create graph to monitor spike events
        print('monitored_neurons model: ', self.monitored_neurons)
        self.graph, self.filter_node, self.sink_node = ut.create_neuron_select_graph(self.model, self.monitored_neurons)

        # Create DYNAP-SE network configuration and apply it to the chip
        print(self.net_gen.network)
        new_config = self.net_gen.make_dynapse1_configuration()
        model.apply_configuration(new_config)
    
        # Create FPGA Spike Generator   
        self.encoding = np.zeros(self.num_spikegens + 3)   
        self.fpga_spike_gen = model.get_fpga_spike_gen()
        
        # Apply DYNAP-SE parameters to the specific cores
        agent_param_group = params.agent_param_group()
        model_param_group = params.model_param_group()
        for i in range(4):
            model.update_parameter_group(agent_param_group, i, 0)
            model.update_parameter_group(agent_param_group, i, 1)
            model.update_parameter_group(model_param_group, i, 2)
            model.update_parameter_group(model_param_group, i, 3)

        
    def init_policy_readout(self):
        # Initialize Adam optimizer for the policy readout
        self.config.read('config.ini')
        alpha = self.config.getfloat('AGENT', 'alpha_rout')
        self.adam_out = Adam(alpha=alpha, drop=0.99, drop_time=10000)
        
        # Initialize policy readout weights
        self.Jout = np.random.normal(0.0, 0.1, size=(self.num_actions, self.num_hidden_neurons))
        self.dJout_aggregate=0
        self.dJout_entropy=0
        self.dJfilt_out = 0
        self.entropy = 0

        self.state_out_policy = np.zeros(self.num_hidden_neurons)
        self.input_policy = np.zeros(self.num_hidden_neurons)
        
    def init_model_readout(self):
        # Initialize Adam optimzers for the state and reward readouts
        alpha_s = self.config.getfloat('MODEL', 'learning_rate_state')
        alpha_r = self.config.getfloat('MODEL', 'learning_rate_reward')
        self.adam_out_s = Adam(alpha=alpha_s, drop=.99, drop_time=10000)
        self.adam_out_r = Adam(alpha=alpha_r, drop=.99, drop_time=10000)
        
        # Initialize state + reward readout weights
        self.Jout_s_pred = np.zeros((4, self.num_hidden_neurons))
        self.Jout_r_pred = np.zeros((1, self.num_hidden_neurons))

        self.dJout_s_aggregate = 0
        self.dJout_r_aggregate = 0
        
        self.state_out_model = np.zeros(self.num_hidden_neurons)
        self.input_model = np.zeros(self.num_hidden_neurons)
        

    def init_policy_net(self):
        # Create spike generators
        self.spike_gens_policy = n.NeuronGroup(0, 0, list(range(self.num_spikegens)), True)
        
        # Create hidden neurons for policy
        neuron_ids_all = list(range(1, self.num_hidden_neurons//2 + 1))
        neuron_group_policy_c0 = n.NeuronGroup(self.target_chip, 0, neuron_ids_all)
        neuron_group_policy_c1 = n.NeuronGroup(self.target_chip, 1, neuron_ids_all)
        
        # Connect Input Neurons to Hidden Neurons
        #### 8 presynaptic connections per postsynaptic neuron (4 weights) ####
        num_connections = 8
        num_weights = 4
        num_pre_neurons = self.num_spikegens
        num_neurons_per_core = self.num_hidden_neurons//2
        prob = num_connections/num_pre_neurons
        arr_i, arr_j = self.set_all_to_all(self.num_spikegens, num_neurons_per_core, p = prob, num_weights=num_weights)
        self.connect_populations(self.net_gen, self.spike_gens_policy.neurons, neuron_group_policy_c0.neurons, arr_i, arr_j, dyn1.Dynapse1SynType.NMDA)
        arr_i, arr_j = self.set_all_to_all(self.num_spikegens, num_neurons_per_core, p = prob, num_weights=num_weights)
        self.connect_populations(self.net_gen, self.spike_gens_policy.neurons, neuron_group_policy_c1.neurons, arr_i, arr_j, dyn1.Dynapse1SynType.NMDA)
            
        # Add neurons to monitor (to record spikes from)
        self.monitored_neurons += neuron_group_policy_c0.tuple_neuron_ids + neuron_group_policy_c1.tuple_neuron_ids
        
    def init_model_net(self):
        # Create spike generators
        self.spike_gens_model_all = n.NeuronGroup(0, 0, list(range(self.num_spikegens + 3)), True)
        
        # Create hidden neurons for model
        neuron_ids_all = list(range(1, self.num_hidden_neurons//2 + 1))
        neuron_group_model_c2 = n.NeuronGroup(self.target_chip, 2, neuron_ids_all)
        neuron_group_model_c3 = n.NeuronGroup(self.target_chip, 3, neuron_ids_all)
        
        # Connect input neurons to hidden neurons
        #### 8 presynaptic connections per postsynaptic neuron (4 weights) ####
        num_connections = 8
        num_weights = 4
        num_pre_neurons = self.num_spikegens
        num_post_neurons = self.num_hidden_neurons//2
        prob = num_connections/num_pre_neurons

        arr_i, arr_j = self.set_all_to_all(num_pre_neurons, num_post_neurons, p = prob, num_weights=num_weights)
        self.connect_populations(self.net_gen, self.spike_gens_model_all.neurons, neuron_group_model_c2.neurons, arr_i, arr_j, dyn1.Dynapse1SynType.NMDA)
        arr_i, arr_j = self.set_all_to_all(num_pre_neurons, num_post_neurons, p = prob, num_weights=num_weights)
        self.connect_populations(self.net_gen, self.spike_gens_model_all.neurons, neuron_group_model_c3.neurons, arr_i, arr_j, dyn1.Dynapse1SynType.NMDA)
        
        # Add neurons to monitor (to record spikes from)
        self.monitored_neurons += neuron_group_model_c2.tuple_neuron_ids + neuron_group_model_c3.tuple_neuron_ids
        
    def set_all_to_all(self, pre_size, post_size, p = 1, num_weights=4):
        output_i = []
        output_j = []
        
        for k in range(post_size):
            num_connections = int(pre_size*p)
            weights = []

            while num_connections > 0:
                # Generate a random smaller number between 1 and num_weights
                weight = random.randint(1, num_weights)
                # Ensure that the weight number does not exceed the remaining num_connections
                weight = min(weight, num_connections)
                # Subtract the weight number from the num_connections
                num_connections -= weight
                # Append the weight number to the list
                weights.append(weight)

            post_idx = sorted(random.sample(range(pre_size), int(len(weights))))

            for m, weight in zip(post_idx, weights):
                for _ in range(weight):
                    output_i.append(int(m))
                    output_j.append(int(k))
        return output_i, output_j
      
    def connect_populations(self, connector, pre_pop, post_pop, i, j, syn_type):
        # Wrapper function for connecting populations given lists of indices
        assert len(i) == len(j), "index lists i and j must be of equal length"
        for index in range(len(i)):
            connector.add_connection(pre_pop[i[index]], post_pop[j[index]], syn_type)

    def step_policy(self, ram):
        # Update agent input
        self.input_next_state_policy(ram)
        
        # Wait 10 ms to process input
        time.sleep(0.01) 
        
        # Read out spike events
        self.get_latest_spike_rates()
        
        # Compute action probabilities and select action
        if self.latest_spike_rates is None: 
            return 0, np.array([0.0,0.0,0.0])
        self.input_policy = np.array(self.latest_spike_rates[:self.num_hidden_neurons])
        self.action, self.probability = self.policy(self.input_policy)
        self.latest_spike_rates[:self.num_hidden_neurons] = np.zeros(self.num_hidden_neurons)
        
        return self.action, self.probability

    def policy(self, state):
        # Compute filtered spike rates
        self.itau_ro = 0.6065306597126334 
        self.state_out_policy = self.state_out_policy * self.itau_ro + state * (1 - self.itau_ro) * 5.0
        
        # Inner product of filtered spike rates and linear readout weights
        y = np.dot(self.Jout, self.state_out_policy)
        y[y>50] = 50
        
        # Compute softmax to get action probabilities
        exp_y = np.exp(y)
        probability = exp_y / np.sum(exp_y)
        
        # Probabilistic action selection
        action = np.random.choice(len(probability), p = probability)

        return int(action), probability
    
    def step_model(self, act_vec, ram):
        # Update model input
        self.input_next_state_model(act_vec, ram)
        
        # Wait 10 ms to process input
        time.sleep(0.01)
        
        # Read out spike events
        self.get_latest_spike_rates()
        if self.latest_spike_rates is None: 
            return 0, np.array([0.0,0.0,0.0])

        # Predict next state and reward
        self.input_model = np.array(self.latest_spike_rates[self.num_hidden_neurons:])
        s_pred, r_pred = self.prediction(self.input_model)
        self.latest_spike_rates[self.num_hidden_neurons:] = np.zeros(self.num_hidden_neurons)

        return s_pred, r_pred

    def prediction(self, state):
        # Compute filtered spike rates
        self.itau_ro = 0.6065306597126334
        self.state_out_model = self.state_out_model * self.itau_ro  + state * (1 - self.itau_ro)
        
        # Compute state + reward prediction
        s_pred = self.Jout_s_pred @ self.state_out_model
        r_pred = self.Jout_r_pred @ self.state_out_model

        return s_pred,r_pred

    def update_policy(self):
        # Update the weights
        self.Jout = self.adam_out.step(self.Jout, self.dJout_aggregate)

        # Clear all values
        self.dJout_aggregate = 0
        self.dJfilt_out = 0
        self.dJout_entropy = 0
        
    def compute_gradient_policy(self, r):
        discount_rate = 0.998
        ac_vector = np.zeros((3,))
        ac_vector[self.action] = 1
        
        # compute the policy entropy
        self.entropy = - np.sum(self.probability*np.log(self.probability))
        
        if self.state_out_policy is None:
            return
        
        # accumulate the policy gradient
        dJ_out =  np.outer((ac_vector - self.probability), self.state_out_policy.T)
        self.dJfilt_out = self.dJfilt_out * discount_rate + dJ_out
        self.dJout_aggregate += (r*self.dJfilt_out)     

    def compute_gradient_model(self, s_pred, r_pred, state, r):
        # compute the model gradients
        self.dJout_s_aggregate += np.outer((state-s_pred), self.state_out_model)
        self.dJout_r_aggregate += np.outer((r-r_pred), self.state_out_model)

    def update_model(self):
        # Update the weights
        self.Jout_s_pred = self.adam_out_s.step(self.Jout_s_pred, self.dJout_s_aggregate)
        self.Jout_r_pred = self.adam_out_r.step(self.Jout_r_pred, self.dJout_r_aggregate)
        
        # Clear all values
        self.dJout_r_aggregate=0
        self.dJout_s_aggregate=0
    
    ##### SNN PART #####
    
    def input_next_state_policy(self, ram):
        # Compute encoding from state values
        self.encoding[:self.num_spikegens] = self.place_cell_encoding(ram)

        # Transform encoding into spike trains and upload them to the chip
        self.set_fpga_spike_gen_rate(self.encoding)
        
    def input_next_state_model(self, act_vec, ram):
        # Compute encoding from state values and chosen action
        encoding = self.place_cell_encoding(ram)
        self.encoding[:] = np.concatenate((encoding, 150*act_vec))

        # Transform encoding into spike trains and upload them to the chip
        self.set_fpga_spike_gen_rate(self.encoding)
        
    def place_cell_encoding(self, ram, sigma=20.0):
        # Create (place cell) activity variables for each input spike generator
        num_place_cells_per_variable = self.num_spikegens//4
        place_cells_ball_x = np.zeros(num_place_cells_per_variable)
        place_cells_ball_y = np.zeros(num_place_cells_per_variable)
        place_cells_cpu_y = np.zeros(num_place_cells_per_variable)
        place_cells_player_y = np.zeros(num_place_cells_per_variable)
        
        # Compute activity for each input spike generator 
        for i in range(num_place_cells_per_variable):
            place_cells_ball_x[i] = self.gaussian(ram[0], i*266/num_place_cells_per_variable - 10, sigma)
            place_cells_ball_y[i] = self.gaussian(ram[1], i*266/num_place_cells_per_variable - 10, sigma)
            place_cells_cpu_y[i] = self.gaussian(ram[2], i*266/num_place_cells_per_variable - 10, sigma)
            place_cells_player_y[i] = self.gaussian(ram[3], i*266/num_place_cells_per_variable - 10, sigma)
        
        scale = 7000
        return scale*np.concatenate((place_cells_ball_x, place_cells_ball_y, place_cells_cpu_y, place_cells_player_y))

    def gaussian(self, x, mu, sigma):
        # Calculate the value of a Gaussian distribution at a given position
        exponent = -((x - mu) ** 2) / (2 * (sigma ** 2))
        coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
        return coefficient * np.exp(exponent)

    def get_latest_spike_rates(self):
        # Read out latest spikes and clear buffer 
        self.latest_spikes = self.sink_node.get_events()

        evts_n = np.array([[evt.timestamp, evt.neuron_id, evt.core_id] for evt in self.latest_spikes])
        
        # Check if there are events
        if len(evts_n) > 0:
            num_events = np.unique(evts_n[:,1] - 1 + 255*evts_n[:,2], return_counts=True)
            self.latest_spike_rates[num_events[0]] += num_events[1]
            return True
        else:
            self.latest_spike_rates = np.zeros((2*self.num_hidden_neurons))
            return False
 
    def set_fpga_spike_gen_rate(self, encoding):
        # Create spike trains
        encoding *= 1.0
        timestamps = np.linspace(0, 1, max(0, int(encoding[0]/33))) / 33
        indices = np.ones(len(timestamps))*self.spikegen_ids[0]
        spiketrain = np.column_stack((timestamps,indices))
        for i in range(1, self.num_spikegens + 3):
            timestamps = np.linspace(0, 1, max(0, int(encoding[i]/33))) / 33
            indices = np.ones(len(timestamps))*self.spikegen_ids[i]
            spiketrain = np.concatenate((spiketrain, np.column_stack((timestamps,indices))))
        spiketrain = spiketrain[spiketrain[:,0].argsort()]

        # Upload the spike trains        
        self.fpga_spike_gen.stop()
        ut.set_fpga_spike_gen(self.fpga_spike_gen, spike_times=spiketrain[:,0], indices=spiketrain[:,1].astype(int), target_chips=[self.target_chip]*len(spiketrain), isi_base=900, repeat_mode=True, is_first=self.is_first)
        if self.is_first:
            self.is_first = False
        self.fpga_spike_gen.start()

        # Plot the spike trains (only for debugging purposes)
        # plt.figure(figsize=(5,18), dpi=150)
        # plt.plot(spiketrain[:,0], spiketrain[:,1], '.')
        # plt.ylim([0, 2*self.num_spikegens+3]) # Set the y-axis limits    
        # plt.xlabel("time (s)")
        # plt.ylabel("Neuron ID")
        # plt.savefig('_spiketrain.png', facecolor='w', edgecolor='w')
        # plt.close()
        # time.sleep(0.2)
        
    def start(self):
        # Start the FPGA spike generator
        self.fpga_spike_gen.start()
        print("FPGA spike gen started")
        self.is_first = True
        
        # Start the graph
        self.graph.start()

        # Clear the buffer 
        self.sink_node.get_events() 

    def stop(self):
        # Stop the FPGA spike generator
        self.fpga_spike_gen.stop()

        # Stop the graph
        self.graph.stop()

    def save(self, filename):
        # Here we collect the relevant quantities to store
        data_bundle = (self.Jout)
        np.save(filename, np.array(data_bundle, dtype=object))
        

   