import random
import numpy as np
import samna.dynapse1 as dyn1
from collections import Counter
import dynapse.dynapse1utils as ut
from dynapse.dynapse1constants import MAX_NUM_CAMS

def weight_matrix2lists(weight_matrix, pre_group, post_group):
    """convert w to pre_list and post_list."""
    if weight_matrix.shape[0] == len(pre_group.neurons) and weight_matrix.shape[1] == len(post_group.neurons):
        pre_list = []
        post_list = []
        for i in range(weight_matrix.shape[0]):
            for j in range(weight_matrix.shape[1]):
                w = weight_matrix[i][j]
                for num_conn in range(int(w)):
                    pre_list.append(i)
                    post_list.append(j)
        return pre_list, post_list
    else:
        raise Exception('Row count of weight_matrix should equals pre_group neuron count, and column count of weight_matrix should equals post_group neuron count!')

def gen_one2one_lists(pre_neurons, post_neurons):
    """Generate one-to-one lists."""
    assert(len(pre_neurons)==len(post_neurons)), 'Pre neuron group '+\
    ' and post neuron group need to have the same length for one2one connections'
    pre_list = np.arange(len(pre_neurons))
    post_list = pre_list
    return pre_list, post_list

def gen_all2all_lists(pre_neurons, post_neurons, p, rand_seed):
    """Generate all-to-all lists."""
    random.seed(rand_seed)

    all2all_conns = []
    for pre in range(len(pre_neurons)):
        for post in range(len(post_neurons)):
            all2all_conns.append((pre, post))
    num_conns = round(len(all2all_conns)*p)
    random_conns = random.sample(all2all_conns, num_conns)

    pre_list = []
    post_list = []
    for pair in random_conns:
        pre_list.append(pair[0])
        post_list.append(pair[1])
    
    random.seed(None)
    return pre_list, post_list

def validate(network, max_num_cams=MAX_NUM_CAMS):
    '''
    Validate if the network meets the DYNAP-SE1 hardware constraits.

    Validation rules:

    First check cams: check the number of incoming connections of each neuron (after reusing cam):
        for each neuron, check its pre neurons:
            if there's pre with same (core_id, neuron_id, synapse_type):
                calculate the number of cams (num_cams) needed by these pre neurons
                num_cams is the maximum weight of between a certain pre and this post
                other pre neurons will also get this max weight cause they are sharing the same cam
                send a warning to the user about 1) the weight sharing 2) different pres sharing same cam
            if the number of incoming connections > 64:
                raise exception

    Then check aliasing: for neurons in the same core, check their pre neurons.
        if two different post neurons have two different pres with
        same (core_id, neuron_id, synapse_type) but not in same chip: raise exception

    No need to check srams
    '''
    valid = False
    # dictionary to save weights > 1, e.g. (pre_tag, post_neuron): num_cams
    post_neuron_dict = network.post_neuron_dict

    for core in post_neuron_dict:
        # get onchip neurons in each core
        post_neurons = post_neuron_dict[core]

        # ----------------------- check cams of all post neurons -----------------------
        for post in post_neurons:
            num_cams = 0
            # pre neurons with different (core_id, neuron_id, synapse_type)
            for pre_tag in post.incoming_connections:
                pre_chip_virtual_list = post.incoming_connections[pre_tag]

                # pre_weight_dict: a dictionary describe the weights between each pre and the post
                # e.g. {(pre1_chip,pre1_is_spike_gen): 1, (pre2_chip,pre2_is_spike_gen): 2}
                # means 1 connection (pre1, post), 2 connections (pre2, post) needed
                # this is an aliasing error:
                # num_cams=1, WRONG for pre2; num_cams=2, WRONG for pre1; num_cams=1, WRONG for both pre1 and pre2
                # If all pre neurons share the same weight, then it's fine
                pre_weight_dict = dict(Counter(pre_chip_virtual_list))
                weight_list = list(pre_weight_dict.values())
                # if all pre neurons have the same weight to the post
                all_pre_same_weight = all(elem == weight_list[0] for elem in weight_list)

                if not all_pre_same_weight:
                    raise Exception("ERROR: aliasing pre neurons exist! Post neuron "+repr(post)+ \
                                    " has pre neurons with same (core_id, neuron_id, synapse_type) " \
                                    +str(pre_tag)+" in different chips. The (pre, post) connections have different weights, which cannot be implemented on chip!")
                weight = weight_list[0]
                # pre neurons with same (core_id, neuron_id, synapse_type) will reuse the same cam of the post neuron
                if len(pre_weight_dict.keys()) > 1:
                    print("WARNING: post neuron "+repr(post)+ \
                        " may have aliasing pre neurons or spike generators with same (core_id, neuron_id, synapse_type) " \
                        +str(pre_tag)+" but in different chips! The (pre, post) connections share weight = "+str(weight)+".")

                # the number of cams needed for this pre tag
                num_cams += weight

            if num_cams > max_num_cams:
                raise Exception("ERROR: post neuron "+repr(post)+" has too many pre neurons or spike generators!")
        # ----------------------- check cams of all post neurons -----------------------

        # ----------------------- check aliasing -----------------------
        # number of post neurons in the same core
        num_post = len(post_neurons)

        # only if there are multiple post neurons in one core, check their pre neurons
        if num_post > 1:
            # compare each neuron with other neurons in the same chip
            # check if they have different pres with same (core_id, neuron_id, synapse_type) but not in same chip
            for nid_1 in range(num_post-1):
                for nid_2 in range(nid_1+1, num_post):
                    pre_dict_1 = post_neurons[nid_1].incoming_connections
                    pre_dict_2 = post_neurons[nid_2].incoming_connections
                    # compare pre neurons with same tag (core_id, neuron_id, synapse_type) cam
                    # of post nid1 and post nid2
                    for pre_tag_1 in pre_dict_1:
                        if pre_tag_1 in pre_dict_2:
                            # pre neuron info (chip_id, spike_gen) list with the same cam should be exact the same
                            # neurons in the same chip, should same neurons/spikegens! otherwise aliasing!
                            # check each pre_chip_item of post1, it should be also in that of post2.

                            pre_chips_1 = sorted(pre_dict_1[pre_tag_1])
                            pre_chips_2 = sorted(pre_dict_2[pre_tag_1])
                            for pre_chip in pre_chips_1:
                                if pre_chip not in pre_chips_2: 

                                    raise Exception("ERROR: aliasing pre neurons exist! Post neurons " \
                                                    +repr(post_neurons[nid_1])+" and " \
                                                    +repr(post_neurons[nid_2]) \
                                                    +" have different pre neurons in different chips but with same (core_id, neuron_id, synapse_type) " \
                                                    +str(pre_tag_1)+"."+" Possible solution: use different pre neuron ids.")
        # ----------------------- check aliasing -----------------------

    # print("Validation complete: the network is good to go!")
    valid = True
    return valid

def convert_validated_network2dynapse1_configuration(network):
    """
    Convert a validated "network" to a Dynapse1Configuration which can be applied using Dynapse1Model.
    """
    # check if the network is empty
    if len(network.post_neuron_dict.keys()) == 0:
        print("WARNING: the network is empty!")

    config = dyn1.Dynapse1Configuration()

    for loc in network.post_neuron_dict:
        # get the chip and core ids of this core
        # for write sram of the pre neurons
        post_chip_id = loc[0]
        post_core_id = loc[1]

        # get onchip neurons in each core
        post_neurons = network.post_neuron_dict[loc]
        for post_neuron in post_neurons:
            post_in_config = ut.get_neuron_from_config(config,
                                        post_chip_id,
                                        post_core_id,
                                        post_neuron.neuron_id)

            for pre_tag in post_neuron.incoming_connections:
                # pre_tag = (pre_neuron.core_id, pre_neuron.neuron_id, synapse_type)
                # for write cam of the post
                pre_core_id = pre_tag[0]
                pre_neuron_id = pre_tag[1]
                synapse_type = pre_tag[2]

                # write cam of this post neuron in configuration
                # this cam serves all the pre neurons under this pre_tag
                # only write cams once for all the pres with the same pre_tag
                # cam reuse is implemented already this way!

                # write how many pre_tags into the cams of post_neuron?
                # num_cams = weights of (pre_tag, post) connection
                # all the pre neurons should have the same weight (validated already) !
                pre_weight_dict = dict(Counter(post_neuron.incoming_connections[pre_tag]))
                weight = list(pre_weight_dict.values())[0]

                check_and_write_post_cam(post_in_config, pre_core_id,
                                        pre_neuron_id, synapse_type,
                                        weight)

                # write the srams of all pre neurons in configuration
                for pre_chip_virtual in post_neuron.incoming_connections[pre_tag]:
                    # pre_chip_virtual = (neuron.chip_id, neuron.is_spike_gen)
                    # if pre is spikeGen, no need to write the sram of the pre.
                    if pre_chip_virtual[1]:
                        continue
                    else:
                        # get the real pre neuron in the configuration
                        pre_in_config = ut.get_neuron_from_config(config,
                                            pre_chip_virtual[0],
                                            pre_core_id,
                                            pre_neuron_id)
                        # write the sram of the pre
                        check_and_write_pre_sram(pre_in_config, post_chip_id, post_core_id)

    return config

def gen_neuron_string(neuron):
    """Warning: gen_neuron_string is deprecated and will be removed in a future release, use str(Neuron) instead.
    """
    print("Warning: gen_neuron_string is deprecated and will be removed in a future release, use str(Neuron) instead.")

    return str(neuron)

def is_same_neuron_loc(neuron1, neuron2):
    """Check if 2 neurons have the same physical ID. Consider a neuron as an individual neuron without 
    any external connections (i.e. not in a Network)

    Args:
        other (netgen.Neuron): a neuron.
    
    Returns:
        True if neuron IDs are the same, False otherwise.
    """
    if neuron1.chip_id == neuron2.chip_id and\
        neuron1.core_id == neuron2.core_id and\
        neuron1.neuron_id == neuron2.neuron_id and\
        neuron1.is_spike_gen == neuron2.is_spike_gen:
        return True
    else:
        return False

def find_neuron_in_dict(neuron, post_neuron_dict):
    """Find neuron in the post_neuron_dict according neuron ID.

    Args:
        neuron (netgen.Neuron): neuron
        post_neuron_dict (dictionary): post_neuron_dict

    Returns:
        netgen.Neuron: neuron in the post_neuron_dict if neuron is in the dictionary,
            None otherwise.
    """
    # if the key exist
    if (neuron.chip_id, neuron.core_id) in post_neuron_dict.keys():
        # if the neuron exists in the corresponding key
        neuron_list = post_neuron_dict[(neuron.chip_id, neuron.core_id)]
        for i in range(len(neuron_list)):
            neuron_in_dict = neuron_list[i]
            if is_same_neuron_loc(neuron, neuron_in_dict):
            # if neuron == neuron_in_dict:
                return i

        return None

def find_pre_in_post_incoming(pre_neuron, post_neuron_in_dict, synapse_type):
    """Find pre in post incoming synapses."""
    pre_dict = post_neuron_in_dict.incoming_connections
    # if the key exist
    if (pre_neuron.core_id, pre_neuron.neuron_id, synapse_type) in pre_dict.keys():
        # if the pre neuron exists in the corresponding key
        pre_neuron_list = pre_dict[(pre_neuron.core_id, pre_neuron.neuron_id, synapse_type)]
        for i in range(len(pre_neuron_list)):
            chip_spikegen_tuple = pre_neuron_list[i]
            # if same neuron, return the first pre_id in the list
            if pre_neuron.chip_id == chip_spikegen_tuple[0] and \
                pre_neuron.is_spike_gen == chip_spikegen_tuple[1]:
                return i
    return None

def write_pre_destination(pre_destinations, pre_dest_id, virtual_core_id, target_chip_id,\
    sx, sy, dx, dy, core_mask, in_use):
    """Write pre destination."""
    pre_destinations[pre_dest_id].virtual_core_id = virtual_core_id
    pre_destinations[pre_dest_id].target_chip_id = target_chip_id
    pre_destinations[pre_dest_id].sx = sx
    pre_destinations[pre_dest_id].sy = sy
    pre_destinations[pre_dest_id].dx = dx
    pre_destinations[pre_dest_id].dy = dy
    pre_destinations[pre_dest_id].core_mask = core_mask
    pre_destinations[pre_dest_id].in_use = in_use

def write_post_synapse(post_synapses, post_synapse_id, listen_core_id, listen_neuron_id, synapse_type):
    """Write post synapse."""
    post_synapses[post_synapse_id].listen_core_id = listen_core_id
    post_synapses[post_synapse_id].listen_neuron_id = listen_neuron_id
    post_synapses[post_synapse_id].syn_type = synapse_type

def get_usable_pre_destination_id(pre_destinations, target_chip_id):
    """
    check if the pre neuron can send a connection to the target chip.
    Attributes:
        pre_destinations: destinations of the pre neuron, sram
        target_chip_id: int
    """
    # check if the pre can send a connection to the target chip
    pre_available = False
    reuse_destination = False

    # if pre already has sram targeting target_chip, no need to write new sram
    pre_dest_id = 0
    for i in range(len(pre_destinations)):
        if pre_destinations[i].target_chip_id == target_chip_id:
            pre_available = True
            pre_dest_id = i
            break

    # if pre does not have sram targeting target_chip, write the first unoccupied sram
    if not pre_available:
        for i in range(len(pre_destinations)):
            if not pre_destinations[i].in_use:
                pre_available = True
                pre_dest_id = i
                break

    if pre_available and pre_destinations[pre_dest_id].in_use:
        reuse_destination = True

    return pre_available, pre_dest_id, reuse_destination

def get_usable_post_synapse_id(post_synapses, listen_core_id, listen_neuron_id, synapse_type, weight):
    """
    check if the post neuron can receive a connection.
    Attributes:
        post_synapses: synapses of the post neuron, cam
        listen_core_id: int, [0,4)
        listen_neuron_id: int, [0,256)
        synapse_type: samna.dynapse1.Dynapse1SynType.NMDA, AMPA, GABA_B, GABA_A
        weight: number of cams needed to be written
    """
    # check if the post can receive a connection
    post_available = False
    post_synapse_id = 0

    # no need to check the reuse because when converting network to configuration
    # we take a "global" view: the pre_tag is brand new to the post everytime this function is called
    # and this function will be called only once which adds the cams for all the pre neurons
    total_cams = len(post_synapses)
    for i in range(total_cams):
        # if there is space left on post_synapses
        # find the first available cams, num_available_cams should = weight
        if (post_synapses[i].listen_neuron_id+\
            post_synapses[i].listen_core_id) == 0 and\
            post_synapses[i].syn_type == dyn1.Dynapse1SynType.NMDA and\
            (total_cams - i) >= weight:
            post_available = True
            post_synapse_id = i
            break

    return post_available, post_synapse_id

def check_and_write_pre_sram(pre, post_chip_id, post_core_id):
    """Check and write pre SRAM.
    """   
    pre_destinations = pre.destinations
    pre_chip_id = pre.chip_id

    pre_available, pre_dest_id, reuse_destination = \
        get_usable_pre_destination_id(pre_destinations, post_chip_id)

    if not pre_available:
        raise Exception("pre neuron has no available outputs!")

    # prepare the coremask: add the new post core into the coremask
    # 1 << post_core_id: shift 1 to left by post_core_id bits
    # 1<<0, 1<<1, 1<<2, 1<<3 = (1, 2, 4, 8) = 0001, 0010, 0100, 1000
    core_mask = pre_destinations[pre_dest_id].core_mask | (1 << post_core_id)

    if reuse_destination:
        # reuse the existing sram, sram already targets the post chip, only need to update the coremask
        pre_destinations[pre_dest_id].core_mask = core_mask

    else:
        # create a new sram for this connection
        virtual_core_id = pre.core_id

        # Routing: calculate the sign and distance between post and pre
        # take the lowest bit as 1 or 0
        # 0&1, 1&1, 2&1, 3&1 = 0000&1, 0001&1, 0010&1, 0011&1 = (0, 1, 0, 1)
        d = (post_chip_id & 1) - (pre_chip_id & 1)
        if d < 0:
            sx = True
        else:
            sx = False
        dx = abs(d)

        # Routing: calculate the sign and distance between post and pre
        # take the second lowest bit as 1 or 0
        # (0&2)>>1, (1&2)>>1, (2&2)>>1, (3&2)>>1 = (0, 0, 1, 1)
        d = ((post_chip_id & 2)>>1) - ((pre_chip_id & 2)>>1)
        if d < 0:
            sy = False
        else:
            sy = True
        dy = abs(d)

        write_pre_destination(pre_destinations, pre_dest_id, virtual_core_id, post_chip_id,\
            sx, sy, dx, dy, core_mask, True)

def check_and_write_post_cam(post, pre_core_id, pre_neuron_id, synapse_type, weight):
    """Check and write post CAM.
    """    
    post_available, post_synapse_id = get_usable_post_synapse_id(post.synapses,
                                                pre_core_id,
                                                pre_neuron_id,
                                                synapse_type,
                                                weight)

    if not post_available:
        raise Exception("post neuron has no available inputs!")

    # write "weight" available cams
    for w in range(weight):
        write_post_synapse(post.synapses, post_synapse_id+w, pre_core_id, pre_neuron_id, synapse_type)