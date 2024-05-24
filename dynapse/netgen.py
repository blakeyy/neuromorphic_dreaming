import samna.dynapse1 as dyn1
import collections
from collections import Counter
import random

import dynapse.dynapse1utils as ut
from dynapse.dynapse1constants import NUM_CHIPS, CORES_PER_CHIP, NEURONS_PER_CORE, MAX_NUM_CAMS
from details.netgen_details import weight_matrix2lists, gen_one2one_lists, gen_all2all_lists, validate, convert_validated_network2dynapse1_configuration, find_neuron_in_dict, find_pre_in_post_incoming

def add_synapses(netgen, synapse):
    """Add a Synapses object into a network generator.

    Args:
        netgen (netgen.NetworkGenerator): NetworkGenerator.
        synapse (netgen.Synapses): synapse that contains pre, post neurons and 
            connections between them.
    """
    for i in range(synapse.mux_conn):
        netgen.add_connections_from_list(synapse.pre.neurons, synapse.post.neurons, synapse.synapse_type, pre_ids=synapse.pre_list, post_ids=synapse.post_list)

def add_wta_conns(netgen, wta_conns):
    """Add WTA connections to create a WTA in the network generator.

    Args:
        netgen (netgen.NetworkGenerator): NetworkGenerator.
        wta_conns (netgen.WTA_connections): WTA_connections that contains excitatory, 
            inhibitory neurons and WTA connections between them.
    """
    add_synapses(netgen, wta_conns.ei)
    add_synapses(netgen, wta_conns.ie)
    if wta_conns.ee is not None:
        add_synapses(netgen, wta_conns.ee)

def remove_synapses(netgen, synapse):
    """Remove a synapse group from a network generator.
    If a neuron pair (pre, post) doesn't have any connections after this 
    remove operation, both of the neurons will be removed from the network generator.

    Args:
        netgen (netgen.NetworkGenerator): NetworkGenerator.
        wta_conns (netgen.synapse): synapse that contains pre, post neurons and 
            connections between them.
    """
    for i in range(synapse.mux_conn):
        netgen.remove_connections_from_list(synapse.pre.neurons, synapse.post.neurons, synapse.synapse_type, pre_ids=synapse.pre_list, post_ids=synapse.post_list)

class Neuron:
    # google docstring style
    """
    Neuron object used to create network topology inside a network generator,
    Not the silicon neuron on DYNAP-SE1 hardware.

    Args:
        chip_id (int): [0,4).Defaults to 0.
        core_id (int): [0,4).Defaults to 0.
        neuron_id (int): [0,256).Defaults to 0.
        is_spike_gen (bool): True if this neuron is a spike generator on the FPGA, otherwise a physical neuron on chip.
            Defaults to False.

    Attributes:
        chip_id (int): Defaults to 0.
        core_id (int): Defaults to 0.
        neuron_id (int): Defaults to 0.
        is_spike_gen (bool): True if this neuron is a spike generator on the FPGA, otherwise a physical neuron on chip.
            Defaults to False.
        incoming_connections (dictionarty): a dictionarty which stores the incoming_connections. Only starts to play a  
            role after add_connection(pre, neuron).
            
            - key (tuple): (pre.core_id,pre.neuron_id,synapse_type). Corresponds to cam. 
                Divide the connections by its  cam value for cam reuse.
            
            - value (list): [(pre.chip_id, pre.is_spike_gen), (pre.chip_id, pre.is_spike_gen),...]. 
                To tell if the post neurons are the same neuron. get the connection weight.
    """
    def __init__(self, chip_id=0, core_id=0, neuron_id=0, is_spike_gen=False):
        if is_spike_gen:
            num_chips = 1
        else:
            num_chips = NUM_CHIPS
        if chip_id >= num_chips or chip_id < 0:
            raise Exception("chip id invalid!")
        if core_id >= CORES_PER_CHIP or core_id < 0:
            raise Exception("core id invalid!")
        if neuron_id >= NEURONS_PER_CORE or neuron_id < 0:
            raise Exception("neuron id invalid!")
        self.chip_id = chip_id
        self.core_id = core_id
        self.neuron_id = neuron_id
        self.is_spike_gen = is_spike_gen
        # (pre.core_id,pre.neuron_id,synapse_type): [(pre.chip_id, pre.is_spike_gen), (pre.chip_id, pre.is_spike_gen),...]
        self.incoming_connections = collections.defaultdict(list)
    
    def __repr__(self):
        """Create official string for class Neuron. print(Neuron), str(Neuron) can be used.
        
        Returns:
            str: string format of a Neuron object.
        """
        if self.is_spike_gen:
            neur_str = 's'
        else:
            neur_str = 'n'

        return f"C{self.chip_id}c{self.core_id}{neur_str}{self.neuron_id}"
    
    def __eq__(self, other):
        """
        Check if 2 neurons have the same physical ID and same incoming connections.

        Args:
            other (netgen.Neuron): a neuron.
        
        Returns:
            True if both neuron IDs and incoming connectionsare the same, 
                False otherwise.
             
        """
        eq_flag1 = self.chip_id == other.chip_id and \
            self.core_id == other.core_id and \
            self.neuron_id == other.neuron_id and \
            self.is_spike_gen == other.is_spike_gen
        
        eq_flag2 = True
        for key in self.incoming_connections:
            eq_flag2 = eq_flag2 and (sorted(self.incoming_connections[key]) == sorted(other.incoming_connections[key]))

        return eq_flag1 and eq_flag2
    
    def __lt__(self, other):
        """	
        Comparison function for sorting. Get called on comparison using < operator.

        Args:
            other (netgen.Neuron): a neuron.
        
        Returns:
            True if this neuron ID is smaller than the other one, False otherwise.
        """
        flag = False
        if self.chip_id < other.chip_id:
            flag = True
        elif self.chip_id == other.chip_id:
            if self.core_id < other.core_id:
                flag = True
            elif self.core_id == other.core_id:
                if self.neuron_id < other.neuron_id:
                    flag = True
        
        return flag
    
    def __hash__(self):
        """	
        Return hashable string of the object.

        Returns:
            str: hashable string of the object.
        """

        h = hash("{}.{}.{}.{}.{}".format(self.chip_id,self.core_id,
                                        self.neuron_id,self.is_spike_gen,
                                        self.incoming_connections))

        return h

class NeuronGroup:
    # numpy docstring style
    """
    A collection of netgen.Neurons for neural population operations. 

    Parameters
    ----------
    chip_id : int
        chip id, defaults to 0.
    core_id : int
        core id, defaults to 0.
    neuron_ids: list[int]
        list of neuron ids, defaults to None.
    is_spike_gen: bool, defaults to False.
        True if this neuron group is a group of spike generators on the FPGA, otherwise silicon neurons on chip.

    Attributes
    ------------
    chip_id : int
        chip id
    core_id : int
        core id 
    neuron_ids: list[int]
        list of neuron ids
    is_spike_gen: bool 
        if this neuron group is a group of spike generators on the FPGA or silicon neurons on chip.

    """
    def __init__(self, chip_id=0, core_id=0, neuron_ids=None, is_spike_gen=False):
        if is_spike_gen:
            num_chips = 1
        else:
            num_chips = NUM_CHIPS

        if chip_id >= num_chips or chip_id < 0:
            raise Exception("chip id invalid!")
        if core_id >= CORES_PER_CHIP or core_id < 0:
            raise Exception("core id invalid!")
        for neuron_id in neuron_ids:
            if neuron_id == None or neuron_id >= NEURONS_PER_CORE or neuron_id < 0:
                raise Exception("neuron ids invalid!")
        
        # check if you use neuron0 of a chip
        if core_id == 0 and (not is_spike_gen):
            for nid in neuron_ids:
                if nid == 0:
                    print("WARNING: be careful, you are using neuron 0 from a chip to construct a neuron group!")

        self.chip_id = chip_id
        self.core_id = core_id
        self.neuron_ids = neuron_ids
        self.is_spike_gen = is_spike_gen

    @property
    def neurons(self):
        """
        neurons getter function to get neurons according to the current chip/core/neuron ids of the neuron group.

        Returns
        -------
        list[netgen.Neuron]
            A list of neurons given the current chip/core/neuron ids.
        """
        neurons = []
        for nid in self.neuron_ids:
            neurons.append(Neuron(self.chip_id,self.core_id,nid,self.is_spike_gen))
        return neurons
    
    @property
    def tuple_neuron_ids(self):
        """list[(chip, core, neuron)] getter function

        Returns
        -------
        list[netgen.Neuron]
            A list of tuple neuron ids given the current chip/core/neuron ids
        """

        tuple_neuron_ids = []
        for nid in self.neuron_ids:
            tuple_neuron_ids.append((self.chip_id,self.core_id,nid))
        return tuple_neuron_ids
    
    def __eq__(self, other):
        """
        Compare two neuron groups. Only compares the neuron ids. 
        Consider a neuron group as an individual group without any external 
        connections (i.e. not in a Network)

        Parameters
        ----------
        other: NeuronGroup
            The other neuron group.
        
        Returns
        -------
        bool
            True if equal, False otherwise.

        """
        return self.chip_id == other.chip_id and \
            self.core_id == other.core_id and \
            self.neuron_ids.any() == other.neuron_ids.any() and \
            self.is_spike_gen == other.is_spike_gen

class Synapses:
    """
    Stores the information of the connectivity from a pre NeuronGroup to a post NeuronGroup.
    The connections can be defined in 3 ways:
    
        - using self-defined pre_list and post_list
        - using connectivity pattern, conn_type
        - using weight_matrix

    Args:
        pre (NeuronGroup): presynaptic neuron group.
        post (NeuronGroup): postsynaptic neuron group.
        synapse_type (samna.dynapse1.Dynapse1SynType): synapse type.
        pre_list (list[int]): the relative index of neurons in the pre_group, not the (chip, core, neuron) id.
            Defaults to None.
        post_list (list[int]): the relative index of neurons in the post_group, not the (chip, core, neuron) id.
            Defaults to None.
        conn_type (str): in ['one2one', 'all2all']. Defaults to None.
        p (float): possibility of all2all random connections. Defaults to None.
        rand_seed (int): random seed of all2all random connections. Defaults to None.
        mux_conn (int): the multiplexer number of the connections. Each pre-post synapse will be mux_conn connections.
            Defaults to 1.

    Attributes:
        pre_group (NeuronGroup): presynaptic neuron group.
        post_group (NeuronGroup): postsynaptic neuron group.
        synapse_type (samna.dynapse1.Dynapse1SynType): synapse type.
        pre_list (list[int]): the relative index of neurons in the pre_group, not the (chip, core, neuron) id.
            Defaults to None.
        post_list (list[int]): the relative index of neurons in the post_group, not the (chip, core, neuron) id.
            Defaults to None.
        conn_type (str): in ['one2one', 'all2all']. Defaults to None.
        p (float): possibility of all2all random connections. Defaults to None.
        rand_seed (int): random seed of all2all random connections. Defaults to None.
        weight_matrix (numpy.array()): weight matrix between pre_group and post_group.
            Rows represent pre_ids, columns represent post_ids. Defaults to None.
        mux_conn (int): the multiplexer number of the connections. Each pre-post synapse will be mux_conn connections.
            Defaults to 1.
    """
    def __init__(self, pre_group, post_group, synapse_type, pre_list=None, post_list=None, conn_type=None, p=None, rand_seed=None, weight_matrix=None, mux_conn=1):
        self.pre = pre_group
        self.post = post_group
        self.synapse_type = synapse_type
        self.mux_conn = mux_conn
        
        # if specify the conn_type, check 
        # 1) if the type is valid 2) if pre_list and post_list are None
        if conn_type != None:
            if (pre_list == None and post_list == None and weight_matrix==None) == False:
                raise Exception('pre_list, post_list or weight_matrix cannot be specified given connection type {conn_type}!')
            
            # generate the pre_list and post_list here, so that user can check which connections are created
            pre_neurons = pre_group.neurons
            post_neurons = post_group.neurons
            if conn_type == 'one2one':
                self.pre_list, self.post_list = gen_one2one_lists(pre_neurons, post_neurons)
            elif conn_type == 'all2all':
                if p == None:
                    p = 1
                self.pre_list, self.post_list = gen_all2all_lists(pre_neurons, post_neurons, p, rand_seed)
            else:
                raise Exception('Connection type not supported!')
            
            self.conn_type = conn_type
            self.p = p
            self.rand_seed = rand_seed
        
        else:
            if pre_list != None and post_list != None and weight_matrix == None:
                assert(len(pre_list)==len(post_list)), 'pre_list '+\
                ' and pre_list need to have the same length'
                self.pre_list, self.post_list = pre_list, post_list
            elif pre_list == None and post_list == None and weight_matrix.any() != None:
                # convert w to pre_list and post_list
                if weight_matrix.shape[0] == len(pre_group.neurons) and weight_matrix.shape[1] == len(post_group.neurons):
                    self.pre_list, self.post_list = weight_matrix2lists(weight_matrix, pre_group, post_group)
            else:
                raise Exception('Please give either (pre_list and post_list) or (weight_matrix)!')

            self.conn_type = None
            self.p = None
            self.rand_seed = None

    def __eq__(self, other):
        """
        Check if 2 Synapses objects are the same, which means same pre and post populations
        and same connections between them.

        Args:
            other (netgen.Synapses): a Synapses object.
        
        Returns:
            True if equal, False otherwise.
             
        """

        if self is None:
            if other is None:
                return True
            else:
                return False
        
        result = (self.pre == other.pre and \
               self.post == other.post and \
               self.synapse_type == other.synapse_type and \
               self.mux_conn == other.mux_conn and \
               self.conn_type == other.conn_type and \
               self.p == other.p and \
               self.rand_seed == other.rand_seed)
        
        if self.pre_list is None:
            result = result and (self.pre_list == other.pre_list)
        else:
            result = result and (list(self.pre_list) == list(other.pre_list))

        if self.post_list is None:
            result = result and (self.post_list == other.post_list)
        else:
            result = result and (list(self.post_list) == list(other.post_list))

        return result

class WTA_connections:
    """Define EI IE EE connections of a WTA with an excitatory and an inhibitory population.

    Attributes:
        ei (netgen.Synapses): EI connections.
        ie (netgen.Synapses): IE connections.
        ee (netgen.Synapses): EE connections. Defaults to None.

    Parameters:
        exc_group (NeuronGroup): excitatory neuron group.
        inh_group (NeuronGroup): inhibitory neuron group.
        syn_type_ei (samna.dynapse1.SynType): EI synapse type.
        syn_type_ie (samna.dynapse1.SynType): IE synapse type.
        syn_type_ee (samna.dynapse1.SynType): recurrent excitatory (EE) synapse type inside      
            excitatory neurons. Defaults to None.
        p_ei (float): probability of EI connections.
        p_ie (float): probability of IE connections.
        ee_pres (list[int]): pre_index_list of the EE connections. Defaults to None.
        ee_posts (list[int]): post_index_list of the EE connections. Defaults to None.
        rand_seed (int): random seed for EI and IE connections. Defaults to None.
        mux_conn_ei (int): multiplexer of EI connections.
        mux_conn_ie (int): multiplexer of IE connections.
        mux_conn_ee (int): multiplexer of EE connections.
    """
    def __init__(self, exc_group, inh_group, syn_type_ei, syn_type_ie, syn_type_ee=None, p_ei=1, p_ie=1, ee_pres=None, ee_posts=None, rand_seed=None, mux_conn_ei=1, mux_conn_ie=1, mux_conn_ee=1):
        self.ei = Synapses(exc_group, inh_group, syn_type_ei, conn_type='all2all', p=p_ei, rand_seed=rand_seed, mux_conn=mux_conn_ei)
        self.ie = Synapses(inh_group, exc_group, syn_type_ie, conn_type='all2all', p=p_ie, rand_seed=rand_seed, mux_conn=mux_conn_ie)

        if syn_type_ee != None:
            if ee_pres == None or ee_posts == None:
                raise Exception('ee_pres and ee_posts must be given to create EE connections')
            self.ee = Synapses(exc_group, exc_group, syn_type_ee, pre_list=ee_pres, post_list=ee_posts, mux_conn=mux_conn_ee)
        else:
            self.ee = None
    
    def __eq__(self, other):
        """
        Check if 2 WTA_connections objects are the same, which means same excitatory and 
        inhibitory populations and same connections between them.

        Args:
            other (netgen.WTA_connections): a WTA_connections object.
        
        Returns:
            True if equal, False otherwise.
             
        """
        return self.ei == other.ei and \
               self.ie == other.ie and \
               self.ee == other.ee

class Network:
    """
    Network that checks and maintains the topology of a spiking neural network that meets DYNAP-SE1 hardware constraints.

    Attributes:
        post_neuron_dict (dictionary): a dictionary which stores all the post neurons
            (and their incoming connections).

            - key: tuple, (post.chip_id, post.core_id).
                Divide the post neurons by its location (core) for aliasing check.

            - value: list of neurons each of which has incoming connections.
    """
    def __init__(self):
        # only track onchip post neurons
        # all connection info already stored in post_neuron.incoming_connections
        self.post_neuron_dict = collections.defaultdict(list)
    
    def __repr__(self):
        """Create official string for class Network. print(Network), str(Network) can be used.
        
        Returns:
            str: string format of a Network object.
        """
        
        if len(self.post_neuron_dict.keys()) == 0:
            return f"The network is empty!"
        else:
            line0 = "Post neuron (ChipId,coreId,neuronId): incoming connections [(preNeuron,synapseType), ...]\n"

            result_str = line0            

            dictionary_items = self.post_neuron_dict.items()
            sorted_items = sorted(dictionary_items)
            for item in sorted_items:
                post_neurons = item[1]
                for post in post_neurons:
                    incoming_connections_list, incoming_connections_str_list = \
                        _convert_incoming_conns_dict2list(post.incoming_connections)
                    result_str += str(post) +': ' + str(incoming_connections_str_list) + '\n'
            
            return f"{result_str}"
    
    def __eq__(self, other):
        """
        Check if 2 Network objects are the same, which means same neurons and same connections between them.

        Args:
            other (netgen.Network): a Network object.
        
        Returns:
            True if equal, False otherwise.
             
        """

        eq_flag = True
        for key in self.post_neuron_dict:
            eq_flag = eq_flag and (sorted(self.post_neuron_dict[key]) == sorted(other.post_neuron_dict[key]))

        return eq_flag

    def add_connection(self, pre, post, synapse_type):
        """Add a connection between the (pre,post) neuron pair and checks if the new connection
        meet DYNAP-SE1 hardware constraints. Raise warnings or exceptions if something goes wrong.

        Args:
            pre (netgen.Neuron): presynaptic neuron.
            post (netgen.Neuron): postsynaptic neuron.
            synapse_type (samna.dynapse1.SynType): synapse type.

        Raises:
            Exception: post neuron cannot be a spike generator.
        """        
        if post.is_spike_gen:
            raise Exception("post neuron cannot be a spike generator!")

        # onchip connections from left chips (i.e. 0,2) to right chips (i.e. 1,3) will make chips die
        if pre.is_spike_gen is False:
            left_chips = [0,2]
            right_chips = [1,3]
            if pre.chip_id in left_chips and post.chip_id in right_chips:
                # raise Exception("connections from left chips [0,2] to right chips [1,3] are forbidden!")
                print("WARNING: you are building connections from left chips [0,2] to right chips [1,3]!")
        
        # Neuron 0 warning. Please avoid neuron 0 of each chip
        if pre.core_id == 0 and pre.neuron_id == 0 and (not pre.is_spike_gen):
            print("WARNING: you are using neuron 0 of chip %i as a pre neuron!" % (pre.chip_id))
        if post.core_id == 0 and post.neuron_id == 0:
            print("WARNING: you are using neuron 0 of chip %i as a post neuron!" % (post.chip_id))

        post_key = (post.chip_id, post.core_id)
        # check if post neuron already in the dict
        # if not, add a post neuron with the pre in its incoming_connections;
        # if yes, only append the pre in post's incoming_connections
        nid_in_list = find_neuron_in_dict(post, self.post_neuron_dict)
        if nid_in_list == None:
            # add a new post neuron with the pre in incoming_conns to the dict
            post.incoming_connections.clear()
            post.incoming_connections[(pre.core_id, pre.neuron_id,
                synapse_type)].append((pre.chip_id, pre.is_spike_gen))
            self.post_neuron_dict[post_key].append(post)
        else:
            # only update the incoming_connections of the post neuron, add the new pre
            self.post_neuron_dict[post_key][nid_in_list].\
                incoming_connections[(pre.core_id, pre.neuron_id,
                synapse_type)].append((pre.chip_id, pre.is_spike_gen))

    def remove_connection(self, pre, post, synapse_type):
        """Remove connection between the (pre,post) neuron pair.
        If the (pre, post) neuron pair only has one connection, the
        neurons will be removed from the network as well. Otherwise
        only this connection is removed. 

        Args:
            pre (netgen.Neuron): presynaptic neuron.
            post (netgen.Neuron): postsynaptic neuron.
            synapse_type (samna.dynapse1.SynType): synapse type.

        Raises:
            Exception: post neuron cannot be a spike generator.
            Exception: connection that does not exist in the network cannot be remove.
            Exception: connection that does not exist in the network cannot be remove.
        """        
        if post.is_spike_gen:
            raise Exception("post neuron cannot be a spike generator!")

        # check if post neuron already in the dict
        post_id_in_list = find_neuron_in_dict(post, self.post_neuron_dict)
        if post_id_in_list == None:
            raise Exception("connection does not exist in the network!")
        else:
            # post neuron in the dict, check if pre in this post's incoming_connections
            post_key = (post.chip_id, post.core_id)
            post_in_dict = self.post_neuron_dict[post_key][post_id_in_list]
            pre_id = find_pre_in_post_incoming(pre, post_in_dict, synapse_type)
            if pre_id == None:
                raise Exception("connection does not exist in the network!")
            else:
                # remove the first pre from the incoming_connections of the post in the dict
                post_incoming_connections_dict = post_in_dict.incoming_connections
                pre_tag = (pre.core_id, pre.neuron_id, synapse_type)
                post_pre_tag_list = post_incoming_connections_dict[pre_tag]

                # remove the pre from the pre_tag list
                post_pre_tag_list.pop(pre_id)
                # self.post_neuron_dict[(post.chip_id, post.core_id)][post_id_in_list].\
                #     incoming_connections[(pre.core_id, pre.neuron_id,
                #     synapse_type)].pop(pre_id)

                # check if incoming_connections.pre_tag becomes empty
                # if empty, remove this pre_tag key from the incoming_connections dict
                if len(post_pre_tag_list) == 0:
                    # self.post_neuron_dict[(post.chip_id, post.core_id)][post_id_in_list].\
                    #     incoming_connections.pop((pre.core_id, pre.neuron_id,
                    #     synapse_type))
                    post_incoming_connections_dict.pop(pre_tag)

                    # check if the entire incoming_connections dict becomes empty
                    # if empty, the post neuron is not a post anymore, remove it from the post_dict[post_key] list
                    if len(post_incoming_connections_dict.keys()) == 0:
                        self.post_neuron_dict[post_key].pop(post_id_in_list)

                        # check if the post_neuron_dict[post_key] list becomes empty
                        # if empty, don't need this post_key anymore, remove this key from post_neuron_dict
                        if len(self.post_neuron_dict[post_key]) == 0:
                            self.post_neuron_dict.pop(post_key)

                            # check if post_neuron_dict becomes empty, send out a message
                            if len(self.post_neuron_dict.keys()) == 0:
                                print("Network cleared!")
    
    def add_connections_from_list(self, pre_neurons, post_neurons, synapse_type, pre_ids, post_ids):
        """Add multiple connections between 2 neuron groups.

        Args:
            pre_neurons (list[netgen.Neuron]): list of neuron objects.
            post_neurons (list[netgen.Neuron]): list of neuron objects.
            synapse_type (samna.dynapse1.SynType): synapse type.
            pre_ids (list[int]): neuron indices in pre_neurons list.
            post_ids (list[int]): neuron indices in post_neurons list.
        """        
        assert(len(pre_ids)==len(post_ids)), 'Pre neuron ids '+\
            ' and post neuron ids need to have the same length'
        
        for (i, j) in zip(pre_ids, post_ids):
            self.add_connection(pre_neurons[i], post_neurons[j], synapse_type)
    
    def remove_connections_from_list(self, pre_neurons, post_neurons, synapse_type, pre_ids, post_ids):
        """Remove multiple connections between 2 neuron groups

        Args:
            pre_neurons (list[netgen.Neuron]): list of neuron objects.
            post_neurons (list[netgen.Neuron]): list of neuron objects.
            synapse_type (samna.dynapse1.SynType): synapse type.
            pre_ids (list[int]): neuron indices in pre_neurons list.
            post_ids (list[int]): neuron indices in post_neurons list.
        """        
        assert(len(pre_ids)==len(post_ids)), 'Pre neuron ids '+\
            ' and post neuron ids need to have the same length'
        
        for (i, j) in zip(pre_ids, post_ids):
            self.remove_connection(pre_neurons[i], post_neurons[j], synapse_type)

    def add_connections_from_type(self, pre_neurons, post_neurons, synapse_type, conn_type, p=1.0, rand_seed=None):
        """Add multiple connections between 2 neuron groups given connectivity type: 'one2one' or 'all2all'.

        Args:
            pre_neurons (list[netgen.Neuron]): list of neuron objects.
            post_neurons (list[netgen.Neuron]): list of neuron objects.
            synapse_type (samna.dynapse1.SynType): synapse type.
            conn_type (str): 'one2one' or 'all2all'.
            p (float, optional): possibility of all2all connections. Defaults to 1.0.
            rand_seed (int, optional): random seed for all2all connections. Defaults to None.
        """        
        print("Warning: this is an old way of creating connections which will not save pre_list and post_list as the connections. The preferable way is to create a Synapses object syn, then use add_synapses(net_gen, syn).")

        if conn_type == 'one2one':
            assert(len(pre_neurons)==len(post_neurons)), 'Pre neuron group '+\
            ' and post neuron group need to have the same length for one2one connections'
            for i in range(len(pre_neurons)):
                self.add_connection(pre_neurons[i], post_neurons[i], synapse_type)
        elif conn_type == 'all2all':
            random.seed(rand_seed)

            all2all_conns = []
            for pre in range(len(pre_neurons)):
                for post in range(len(post_neurons)):
                    all2all_conns.append((pre, post))
            num_conns = round(len(all2all_conns)*p)
            random_conns = random.sample(all2all_conns, num_conns)
            for pair in random_conns:
                pre_id = pair[0]
                post_id = pair[1]
                self.add_connection(pre_neurons[pre_id], post_neurons[post_id], synapse_type)
            
            random.seed(None)

class NetworkGenerator:
    """NetworkGenerator maintains a Network and a corresponding Dynapse1Configuration.
    A NetworkGenerator must be used to convert you network topology added to the
    NetworkGenerator into a Dynapse1Configuration that can be applied to hardware.

    Whenever you want to change your network, you need to add/remove connections
    using the previous NetworkGenerator from which you took the configuration
    and applied it using the model during your last modification of the network.
    Otherwise, you just create a new NetworkGenerator and configuration
    from the scratch and apply it.

    Attributes:
        config (samna.dynapse1.Dynapse1Configuration): hardware applicable DYNAP-SE1 configuration converted
            from the network stored in the NetworkGenerator.
        network (netgen.Network): network description of neurons and connections.

    """    
    def __init__(self):
        self.config = dyn1.Dynapse1Configuration()
        self.network = Network()

    def add_connection(self, pre, post, synapse_type):
        """Add a connection between the pre and post neuron with type "synapse_type" into the network.

        Args:
            pre (netgen.Neuron): presynaptic neuron.
            post (netgen.Neuron): postsynaptic neuron.
            synapse_type (samna.dynapse1.SynType): synapse type.
        """        
        self.network.add_connection(pre, post, synapse_type)

    def remove_connection(self, pre, post, synapse_type):
        """Remove a connection between the pre and post neuron with type "synapse_type" into the network.

        Args:
            pre (netgen.Neuron): presynaptic neuron.
            post (netgen.Neuron): postsynaptic neuron.
            synapse_type (samna.dynapse1.SynType): synapse type.

        """    
        self.network.remove_connection(pre, post, synapse_type)
    
    def add_connections_from_list(self, pre_neurons, post_neurons, synapse_type, pre_ids, post_ids):
        """Add multiple connections between 2 neuron groups.

        Args:
            pre_neurons (list[netgen.Neuron]): list of neuron objects.
            post_neurons (list[netgen.Neuron]): list of neuron objects.
            synapse_type (samna.dynapse1.SynType): synapse type.
            pre_ids (list[int]): neuron indices in pre_neurons list.
            post_ids (list[int]): neuron indices in post_neurons list.
        """    
        self.network.add_connections_from_list(pre_neurons, post_neurons, synapse_type, pre_ids, post_ids)
    
    def remove_connections_from_list(self, pre_neurons, post_neurons, synapse_type, pre_ids, post_ids):
        """Remove multiple connections between 2 neuron groups

        Args:
            pre_neurons (list[netgen.Neuron]): list of neuron objects.
            post_neurons (list[netgen.Neuron]): list of neuron objects.
            synapse_type (samna.dynapse1.SynType): synapse type.
            pre_ids (list[int]): neuron indices in pre_neurons list.
            post_ids (list[int]): neuron indices in post_neurons list.
        """   
        self.network.remove_connections_from_list(pre_neurons, post_neurons, synapse_type, pre_ids, post_ids)
    
    def add_connections_from_type(self, pre_neurons, post_neurons, synapse_type, conn_type, p=1, rand_seed=None):
        """Add multiple connections between 2 neuron groups given connectivity type: 'one2one' or 'all2all'.

        Args:
            pre_neurons (list[netgen.Neuron]): list of neuron objects.
            post_neurons (list[netgen.Neuron]): list of neuron objects.
            synapse_type (samna.dynapse1.SynType): synapse type.
            conn_type (str): 'one2one' or 'all2all'.
            p (float, optional): possibility of all2all connections. Defaults to 1.0.
            rand_seed (int, optional): random seed for all2all connections. Defaults to None.
        """
        self.network.add_connections_from_type(pre_neurons, post_neurons, synapse_type, conn_type, p, rand_seed)

    def clear_network(self):
        """
        Clear all neurons and connections added to the network.
        """
        self.network.post_neuron_dict.clear()

    def print_network(self):
        """Warning: print_network is deprecated and will be removed in a future release,
            use print(NetworkGenerator.network). Note: str(NetworkGenerator.network) gives
            you the string format of a network.
        """

        print("Warning: print_network is deprecated and will be removed in a future release, \
            use print(NetworkGenerator.network). Note: str(NetworkGenerator.network) gives \
            you the string format of a network.")

        print(self.network)

    def make_dynapse1_configuration(self, validation=True):
        """Convert network defined in the network generator into a DYNAP-SE1 applicable configuration.
        Checks if the network of the network generator is valid or not,
        i.e. meets DYNAP-SE1 hardware constraints.
        If valid, network will be converted to a DYNAP-SE1 configuration.
        Otherwise, exceptions will be raised.

        Args:
            validation (bool, optional): whether to validate the network before converting
                to a DYNAP-SE1 configuration. Defaults to True.

        Returns:
            samna.dynapse1.Dynapse1Configuration: Dynapse1Configuration.
        """        
        if validation:
            is_valid = validate(self.network, MAX_NUM_CAMS)

        self.config = convert_validated_network2dynapse1_configuration(self.network)
        # print("Converted the validated network to a Dynapse1 configuration!")
        
        return self.config

    def make_dynapse1_configuration_in_chip(self, chip_id):
        """Convert network defined in the network generator into a DYNAP-SE1 applicable configuration
        which only lies in one target chip. The configuration of the remaining 3 chips will not be changed.

        This function should be followed by :func:`apply_configuration_by_chip() <Dynapse1Model.apply_configuration_by_chip>` which applies the new configuration only to a single chip of the DYNAP-SE1 board. 

        NOTE:
            Be careful if you use this method because this is specifically designed for the user who only does experiments in a single chip. If you have network components in other chips, you will get an error. 

        Args:
            chip_id (int): chip id, [0,4).

        Raises:
            Exception: network has neuron(s) outside the target chip.

        Returns:
            samna.dynapse1.Dynapse1Configuration: Dynapse1Configuration.
        """        
        # first check if the neurons in the network are all in the chip
        post_neuron_dict = self.network.post_neuron_dict
        for post_chip_core in post_neuron_dict:
            # check chip id of the post neuron
            if post_chip_core[0] != chip_id:
                raise Exception("ERROR: network has neuron(s) outside chip "+str(chip_id)+"!")

            # check the chip ids of all the pre neurons of the post neurons
            post_neurons = post_neuron_dict[post_chip_core]
            for post_neuron in post_neurons:
                for pre_tag in post_neuron.incoming_connections:
                    pre_chip_spikegens = post_neuron.incoming_connections[pre_tag]
                    for pre_chip_spikegen in pre_chip_spikegens:
                        if pre_chip_spikegen[1] == False and pre_chip_spikegen[0] != chip_id:
                            raise Exception("ERROR: network has neuron(s) outside chip "+str(chip_id)+"!")

        print("Neurons in the network are all located in chip "+str(chip_id)+".")

        config = self.make_dynapse1_configuration()
        return config

    def make_dynapse1_configuration_in_core(self, chip_id, core_id):
        """Convert network defined in the network generator into a DYNAP-SE1 applicable configuration
        which only lies in one target core. The configuration of the remaining 15 cores will not be changed.

        This function should be followed by :func:`apply_configuration_by_core() <Dynapse1Model.apply_configuration_by_core>` which applies the new configuration only to a single chip of the DYNAP-SE1 board.
        
        NOTE: 
            Be careful if you use this method because this is specifically designed for the user who only does experiments in a single core. If you have network components in other core, you will get an error.

        Args:
            chip_id (int): chip id, [0,4).
            core_id (int): core id, [0,4).

        Raises:
            Exception: network has neuron(s) outside the target core.

        Returns:
            samna.dynapse1.Dynapse1Configuration: Dynapse1Configuration.
        """
        # first check if the neurons in the network are all in the core
        post_neuron_dict = self.network.post_neuron_dict
        for post_chip_core in post_neuron_dict:
            # check core and chip id of the post neuron
            if post_chip_core[0] != chip_id or post_chip_core[1] != core_id:
                raise Exception("ERROR: network has neuron(s) outside chip "
                                +str(chip_id)+", core "+str(core_id)+"!")

            # check the core and chip ids of all the pre neurons of the post neurons
            post_neurons = post_neuron_dict[post_chip_core]
            for post_neuron in post_neurons:
                for pre_tag in post_neuron.incoming_connections:
                    pre_chip_spikegens = post_neuron.incoming_connections[pre_tag]
                    for pre_chip_spikegen in pre_chip_spikegens:
                        if pre_chip_spikegen[1] == False:
                            # for all the physical neurons, check their core and chip ids.
                            chip = pre_chip_spikegen[0]
                            core = pre_tag[0]
                            if chip != chip_id or core != core_id:
                                raise Exception("ERROR: network has neuron(s) outside chip "
                                                +str(chip_id)+", core "+str(core_id)+"!")

        print("Neurons in the network are all located in chip "+str(chip_id)+", core "+str(core_id)+".")

        config = self.make_dynapse1_configuration()
        return config

def _convert_incoming_conns_dict2list(incoming_connections_dict):
    """
    Convert (pre.core_id,pre.neuron_id,synapse_type): [(pre.chip_id, pre.is_spike_gen), (pre.chip_id, pre.is_spike_gen),...]
    to
    [(preNeuron1,synapseType1), (preNeuron1,synapseType2), (preNeuron1,synapseType2), (preNeuron2,synapseType1),...]
    """
    incoming_connections_list = []
    incoming_connections_str_list = []

    for pre_tag in incoming_connections_dict:
        core = pre_tag[0]
        neuron = pre_tag[1]
        syn_type = pre_tag[2]
        if syn_type == dyn1.Dynapse1SynType.NMDA:
            syn_str = "NMDA"
        elif syn_type == dyn1.Dynapse1SynType.AMPA:
            syn_str = "AMPA"
        elif syn_type == dyn1.Dynapse1SynType.GABA_B:
            syn_str = "GABA_B"
        elif syn_type == dyn1.Dynapse1SynType.GABA_A:
            syn_str = "GABA_A"

        pre_neurons = incoming_connections_dict[pre_tag]
        
        for pre_neuron in pre_neurons:
            chip = pre_neuron[0]
            is_spike_gen = pre_neuron[1]
            pre_neuron = Neuron(chip,core,neuron,is_spike_gen)
            incoming_connections_list.append((pre_neuron,syn_type))
            incoming_connections_str_list.append((repr(pre_neuron),syn_str))

    return incoming_connections_list, incoming_connections_str_list
    
def print_post_neuron_dict(post_neuron_dict):
    """Print post_neuron_dict of the Network.

    Args:
        post_neuron_dict (dictionary): post_neuron_dict.
    """    
    if len(post_neuron_dict.keys()) == 0:
        print("The network is empty!")
    else:
        print("Post neuron (ChipId,coreId,neuronId): incoming connections [(preNeuron,synapseType), ...]")
        dictionary_items = post_neuron_dict.items()
        sorted_items = sorted(dictionary_items)
        for item in sorted_items:
            post_neurons = item[1]
            for post in post_neurons:
                incoming_connections_list, incoming_connections_str_list = \
                    _convert_incoming_conns_dict2list(post.incoming_connections)
                print(post+":", incoming_connections_str_list)