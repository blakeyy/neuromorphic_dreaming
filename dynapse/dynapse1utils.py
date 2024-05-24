import samna
import samna.dynapse1 as dyn1
import socket
from dynapse.dynapse1constants import *
import random
import time
import json
from threading import Thread
import numpy as np
import sys, os
import warnings
import re

def free_port():
    """
    Get a free port number using sockets.
    """
    free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_socket.bind(('0.0.0.0', 0))
    free_socket.listen(5)
    port = free_socket.getsockname()[1]
    free_socket.close()
    return port

def open_device(sender_port=33336, receiver_port=33335, select_device=False):
    """
    Get unopened devices detected by samna.

    Args:
        sender_port (int): samnaNode's sending port.
        receiver_port (int): samnaNode's receiving port.
        select_device (bool): Defaults to False.
    
    Returns:
        device: Samna device model
        samna_info_dict (dictionary): information dictionary.
    """
    # ----------- connect Python to C++ ----------------
    sender_endpoint = "tcp://0.0.0.0:"+str(sender_port)
    receiver_endpoint = "tcp://0.0.0.0:"+str(receiver_port)
    node_id = 1
    interpreter_id = 2
    # note: has to assign to "samna_node",
    # otherwise "RuntimeError: Store with ID: 1 timed out on content request"
    try:
        global samna_node
        samna_node = samna.SamnaNode(sender_endpoint, receiver_endpoint, node_id)
    except Exception as e:
        print("ERROR: "+str(e)+", please re-run open_device()!")

    # setup the python interpreter node
    samna.setup_local_node(receiver_endpoint, sender_endpoint, interpreter_id)
    # open a connection to device_node, i.e. the store
    samna.open_remote_node(node_id, "device_node")
    # ----------- connect Python to C++ ----------------

    # retrieve unopened device
    devices = samna.device.get_unopened_devices()

    if len(devices) == 0:
        raise Exception("no device detected!")

    # let user select a device to open
    for i in range(len(devices)):
        print("["+str(i)+"]: ", devices[i], "serial_number", devices[i].serial_number)

    if select_device:
        idx = input("Select the device you want to open by index: ")
    else:
        idx = 0

    # open the device
    # print("idx:", idx)
    # print("device 0:", devices[int(0)])
    # print("device 1:", devices[int(1)])
    # print("device 2:", devices[int(2)])
    device = samna.device.open_device(devices[int(idx)])
    # print("device:", device)
    samna_info_dict = {
        "sender_port":sender_endpoint,
        "receiver_port":receiver_endpoint,
        "samna_node_id":node_id,
        "device_name":devices[int(idx)].device_type_name,
        "python_node_id":interpreter_id
    }

    return device, samna_info_dict

def open_dynapse1(gui=False, select_device=False, sender_port=33337, receiver_port=33338):
    """
    open DYNAP-SE1 board with or without GUI. Note that input parameter device_name 
    has been deprecated now, please do not assign device_name anymore! 
    You cannot name DYNAP-SE1 board by yourself because it's now assigned as 
    'Dynapse1DevKit:index' by open_dynapse1()

    :param bool gui: whether to open the gui or not. "True" will return store and gui_process;
        "False" will only return store.
    :param bool select_device: whether to select one DYNAP-SE1 board out of some connected ones.
        "False": board 0 will be opened; "True": user will be asked to choose the board index
    :param int sender_port: samnaNode's sending port. Should be 33336 if gui=True, 
        otherwise you can use a 5-digits port number.
    :param int receiver_port: samnaNode's receiving port. Should be 33335 if gui=True, 
        otherwise you can use a 5-digits port number which is different from sender_port.

    :returns:
        Dynapse1Model: model, DYNAP-SE1 model of the selected device.
        gui_process: the gui process handler if gui=True. Otherwise, an empty string ''.

    """
    # ports = random.sample(range(10**4, 10**5), k=2)

    if gui:
        # has to be these 2 numbers if you want to run the GUI
        sender_port=33336
        receiver_port=33335
    
    model, samna_info_dict = open_device(sender_port, receiver_port, select_device)
    
    if gui:
        visualizer_id = 3

        # open the gui
        gui_process, gui_receiving_port = open_gui(model, visualizer_id)

        samna_info_dict["gui_receiving_port"] = gui_receiving_port
        samna_info_dict["gui_node_id"] = visualizer_id
        print("GUI receiving port:", samna_info_dict["gui_receiving_port"])
        print("GUI node ID:", samna_info_dict["gui_node_id"])

    print("Sender port:", samna_info_dict["sender_port"])
    print("Receiver port:", samna_info_dict["receiver_port"])
    print("Opened device name:", samna_info_dict["device_name"])
    print("SamnaNode ID:", samna_info_dict["samna_node_id"])
    print("PythonNode ID:", samna_info_dict["python_node_id"])
    
    with open('samna_info.json', 'w') as json_file:
        json.dump(samna_info_dict, json_file, indent=4)

    if gui:
        return model, gui_process
    else:
        return model, ''

def open_gui(device, visualizer_id=3):
    """Open DYNAP-SE1 spiking GUI.
    """
    # add a node in filter gui_graph
    global gui_graph
    gui_graph = samna.graph.EventFilterGraph()
    # Add a converter node that translate the raw DVS events
    dynapse1_to_visualizer_converter_node = get_Dynapse1_viz_converter() # Dynapse1EventToVizConverter
    # Add a streamer node that streams visualization events to our graph
    
    source_node, converter_node, streamer_node = gui_graph.sequential([device.get_source_node(), 
    dynapse1_to_visualizer_converter_node, "VizEventStreamer"])

    gui_graph.start()

    # create gui process
    visualizer, gui_thread = open_visualizer(0.75, 0.75, samna_node.get_receiver_endpoint(), 
    samna_node.get_sender_endpoint(), visualizer_id)

    port = random.randint(10**4, 10**5)
    viz_name = "visualizer"+str(visualizer_id)
    # open a connection to the GUI node
    visualizer = getattr(samna, viz_name)

    try:
        # The GUI node contains a ZMQ receiver endpoint by default, we can set the address it should listen on
        gui_receiving_port = "tcp://0.0.0.0:"+str(port)
        visualizer.receiver.set_receiver_endpoint(gui_receiving_port) # local connection on port 40000
    except Exception as e:
        print("ERROR: "+str(e)+", please re-run open_gui()!")

    # stream on the same endpoint as the receiver is listening to
    streamer_node.set_streamer_endpoint(gui_receiving_port)

    # Connect the receiver output to the visualizer plots input
    visualizer.receiver.add_destination(visualizer.splitter.get_input_channel())

    # Add plots to the GUI
    activity_plot_id = visualizer.plots.add_activity_plot(64, 64, "DYNAP-SE1")
    visualizer.splitter.add_destination("passthrough", visualizer.plots.get_plot_input(activity_plot_id))

    # List currently displayed plots
    visualizer.plots.report()

    return gui_thread, gui_receiving_port

def get_Dynapse1_viz_converter():
    """Helper function to reuse viz node for DYNAP-SE1 GUI.
    """    
    filter_source = '''
        template<typename T>
        class Dynapse1ToViz : public iris::FilterInterface<std::shared_ptr<const std::vector<T>>, std::shared_ptr<const std::vector<ui::Event>>> {
        public:
            void apply() override
            {
                auto outputCollection = std::make_shared<std::vector<ui::Event>>();
                while (const auto input = this->receiveInput()) {
                    std::for_each(input.value()->cbegin(), input.value()->cend(),
                                  [&](auto&& ev) {
                                      if (std::holds_alternative<dynapse1::Spike>(ev)) {
                                          const dynapse1::Spike& spike = std::get<dynapse1::Spike>(ev);
                                          uint16_t row, col, core_row, core_col;
                                          core_row = 16;
                                          core_col = 16;
    
                                          row = spike.chipId / 2 * core_row * 2 +
                                                spike.coreId / 2 * core_row +
                                                spike.neuronId / core_row;
                                          col = spike.chipId % 2 * core_col * 2 +
                                                spike.coreId % 2 * core_col +
                                                spike.neuronId % core_col;
    
                                          uint32_t polarity;
    
                                          if (static_cast<uint16_t>(spike.coreId) == 0 or static_cast<uint16_t>(spike.coreId) == 3) {
                                              polarity = 1;
                                          }
                                          else {
                                              polarity = 0;
                                          }
    
                                          outputCollection->emplace_back(ui::DvsEvent{0,
                                                                         63 - static_cast<uint64_t>(row),
                                                                         static_cast<uint64_t>(col),
                                                                         0, static_cast<uint32_t>(spike.timestamp),
                                                                         polarity});
                                      }
                                  });
                }
                if (outputCollection->empty()) {
                    return;
                }
    
                this->forwardResult(std::move(outputCollection));
            }
    
        };
    
        namespace svejs {
        template<typename T>
        struct RegisterImplementation<Dynapse1ToViz<T>> {
    
            using Type = Dynapse1ToViz<T>;
    
            static constexpr inline auto registerConstructors()
            {
                return constructors(constructor<>());
            }
    
            static constexpr inline auto registerMembers()
            {
                return members();
            }
    
            static constexpr inline auto registerMemberFunctions()
            {
                return memberFunctions();
            }
    
            static constexpr inline auto registerBaseClasses()
            {
                return svejs::BaseClasses<iris::NodeInterface>();
            }
    
            static inline auto registerName()
            {
                return std::string("Dynapse1ToViz_") + svejs::snakeCase(svejs::registerName<T>());
            }
        };
        }
    '''
    
    return samna.graph.JitFilter('Dynapse1ToViz', filter_source)

def open_visualizer(window_width, window_height, receiver_endpoint, sender_endpoint, visualizer_id):
    """start visualizer in a isolated process which is required on mac, intead of a sub process.
    It will not return until the remote node is opened. Return the opened visualizer.
    """
    gui_cmd = f"import samna, samnagui; samnagui.runVisualizer({window_width}, {window_height}, \
        '{receiver_endpoint}', '{sender_endpoint}', {visualizer_id})"
    os_cmd = f'{sys.executable} -c "{gui_cmd}"'
    print("Visualizer start command: ", os_cmd)
    gui_thread = Thread(target=os.system, args=(os_cmd,))
    gui_thread.start()

    # wait for open visualizer and connect to it.
    timeout = 10
    begin = time.time()
    name = "visualizer" + str(visualizer_id)
    while time.time() - begin < timeout:
        try:
            time.sleep(0.05)
            samna.open_remote_node(visualizer_id, name)
        except:
            continue
        else:
            return getattr(samna, name), gui_thread

    raise Exception("open_remote_node failed:  visualizer id %d can't be opened in %d seconds!!" 
    % (visualizer_id, timeout))


def close_dynapse1(model, gui_process=''):
    '''
    Close DYNAP-SE1 board with or without the GUI.

    :param Dynapse1Model model: the DYNAP-SE1 model you get when you 
        :func:`open_dynapse1 <dynapse1utils.open_dynapse1>`.

    :param Process gui_process: the GUI process handler you created when you 
        :func:`open_dynapse1 <dynapse1utils.open_dynapse1>`.

    '''
    if gui_process != '':
        gui_process.join()
    samna.device.close_device(model)

def get_neuron_from_config(config, chip, core, neuron):
    """
    Get a neuron by its global_neuron_id from a configuration.

    :param Dynapse1Configuration config: Dynapse1Configuration. 
    :param int global_neuron_id: global neuron id, [0,4096).

    :returns: Dynapse1Neuron.
    :rtype: samna.dynapse1.Dynapse1Neuron().
    """
    return config.chips[chip].cores[core].neurons[neuron]

def gen_synapse_string(synapse):
    """Convert a Dynapse1Synapse in the string format for printing: 
    c<listen_core_id>n<listen_neuron_id><synapse_type>. 
    e.g., c1n25NMDA means this synapse listens to core 1 neuron 25 with type NMDA.

    :param Dynapse1Synapse synapse: synapse.
    """
    syn_type = synapse.syn_type
    if syn_type == dyn1.Dynapse1SynType.NMDA:
        syn_str = "NMDA"
    elif syn_type == dyn1.Dynapse1SynType.AMPA:
        syn_str = "AMPA"
    elif syn_type == dyn1.Dynapse1SynType.GABA_B:
        syn_str = "GABA_B"
    elif syn_type == dyn1.Dynapse1SynType.GABA_A:
        syn_str = "GABA_A"

    return "c"+str(synapse.listen_core_id)+\
            "n"+str(synapse.listen_neuron_id)+\
            syn_str

def print_neuron_synapses(neuron, synapse_id_list=range(MAX_NUM_CAMS)):
    """Print the synapses of the neuron given the synapse id list.

    :param Dynapse1Neuron neuron: neuron.
    :param list[int] synapse_id_list: synapse IDs you want to print out.
    """
    synapses = neuron.synapses
    for i in synapse_id_list:
        if i == len(synapse_id_list)-1:
            end = '\n'
        else:
            end = ','
        print(gen_synapse_string(synapses[i]), end = end)

def gen_destination_string(destination):
    """Convert a Dynapse1Destination in the string format for printing: 
    C<target_chip_id>c<core_mask><in_use>. 
    e.g., C2c1True means this Dynapse1Destination has target_chip_id=2, 
    core_mask=1 and it's occupied.

    :param Dynapse1Destination destination: destination.
    """
    return "C"+str(destination.target_chip_id)+\
            "c"+str(destination.core_mask)+\
            str(destination.in_use)

def print_neuron_destinations(neuron, destination_id_list=range(4)):
    """Print the destinations of the neuron given the destination id list. 

    :param Dynapse1Neuron neuron: neuron.
    :param list[int] destination_id_list: destination IDs you want to print out.

    """
    destinations = neuron.destinations
    for i in destination_id_list:
        if i == len(destination_id_list)-1:
            end = '\n'
        else:
            end = ','
        print(gen_destination_string(destinations[i]), end = end)

def get_global_id(chip, core, neuron):
    """Conveniently generate the corresponding global neuron ID given chip, core, neuron IDs.

    :param int chip: chip ID, [0,4).
    :param int core: core ID, [0,4) for physical neurons, [0,1) for spike generators on the FPGA.
    :param int neuron: neuron ID, [0,256).

    :returns: global_neuron_id.
    :rtype: int.
    """
    if (isinstance(chip, int) and isinstance(core, int) and isinstance(neuron, int)) == False:
        raise Exception("Chip core and neuron IDs should be integer!")
    return neuron+core*NEURONS_PER_CORE+chip*NEURONS_PER_CHIP

def get_global_id_list(tuple_list):
    """Conveniently generate a list of global neuron IDs given a list of [chip, core, neuron ID] tuples.

    :param list[[int,int,int]] tuple_list: a list of global neuron IDs.

    :returns: global_neuron_ids.
    :rtype: list[int].
    """
    return [tuple_list[2]+tuple_list[1]*NEURONS_PER_CORE+tuple_list[0]*NEURONS_PER_CHIP\
                for tuple_list in tuple_list]

def get_parameters(config, chip, core):
    """Get a list of Dynapse1Parameter (25 in total) of a specific core.

    :param Dynapse1Configuration config: Dynapse1Configuration retrieved from Dynapse1Model.
    :param int chip: chip ID, [0,4).
    :param int core: core ID, [0,4).

    :returns: current parameter setup of the core.
    :rtype: list[samna.dynapse1.Dynapse1Parameter].
    """
    return config.chips[chip].cores[core].parameter_group.param_map.values()

def get_parameter_value(config, chip, core, name):
    """Get the parameter (coarse, fine) value of a specific parameter in the
    DYNAP-SE1 configuration.

    Args:
        config (samna.dynapse1.Dynapse1Configuration): Dynapse1Configuration
        chip (int): chip ID.
        core (int): core ID.
        name (str): parameter name.

    Returns:
        tuple(int,int): (coarse, fine) value.
    """
    return (config.chips[chip].cores[core].parameter_group.param_map[name].coarse_value, 
    config.chips[chip].cores[core].parameter_group.param_map[name].fine_value)

def save_parameters2txt_file(config, filename="./dynapse_parameters.txt"):
    """Save the parameters of 16 cores into a txt file.

    :param Dynapse1Configuration config: Dynapse1Configuration retrieved from Dynapse1Model.
    :param string filename: the path and filename of the output paramter file.
    """
    save_file = open(filename, "w")
    for chip in range(NUM_CHIPS):
        for core in range(CORES_PER_CHIP):
            params = get_parameters(config, chip, core)
            for param in params:
                save_file.write(
                    'C{0}c{1}:({2},{3},{4})\n'.format(
                        chip,
                        core,
                        param.param_name,
                        param.coarse_value,
                        param.fine_value))

def set_parameters_in_txt_file(model, filename="./dynapse_parameters.txt"):
    """Load the parameters in a txt file to DYNAP-SE1 board.

    :param Dynapse1Model model: Dynapse1Model.
    :param string filename: the path and filename of the input paramter file.
    """
    # parse file and update parameters
    with open(filename) as f:
        lines = f.read().splitlines()

    non_empty_lines = [line for line in lines if line.strip() != ""]

    # Strips the newline character
    for line in non_empty_lines:
        chip = line.split('C')[1].split('c')[0].strip()
        core = line.split('c')[1].split(':')[0].strip()
        name = line.split('(')[1].split(',')[0].strip()
        coarse_value = line.split(',')[1].strip()
        fine_value = line.split(',')[2].split(')')[0].strip()

        param = dyn1.Dynapse1Parameter(name, int(coarse_value), int(fine_value))

        model.update_single_parameter(param, int(chip), int(core))

def save_parameters2json_file(config, filename="./dynapse_parameters.json"):
    """Save the parameters of 16 cores into a JSON file.

    :param Dynapse1Configuration config: Dynapse1Configuration retrieved from Dynapse1Model.
    :param string filename: the path and filename of the output paramter file.

    """
    data = {}
    data['parameters'] = []
    for chip in range(NUM_CHIPS):
        for core in range(CORES_PER_CHIP):
            params = get_parameters(config, chip, core)
            for param in params:
                param_dict = {
                    "chip":chip,
                    "core":core,
                    "parameter_name":param.param_name,
                    "coarse_value":param.coarse_value,
                    "fine_value":param.fine_value
                }
                data['parameters'].append(param_dict)

    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def set_parameters_in_json_file(model, filename="./dynapse_parameters.json"):
    """Load the parameters in a JSON file to DYNAP-SE1 board.

    :param Dynapse1Model model: Dynapse1Model.
    :param string filename: the path and filename of the input paramter file.
    """
    with open(filename) as json_file:
        data = json.load(json_file)
    for p in data['parameters']:
        param = dyn1.Dynapse1Parameter(p['parameter_name'], int(p['coarse_value']), int(p['fine_value']))
        model.update_single_parameter(param, int(p['chip']), int(p['core']))

def get_serial_number(device_idx=0, select_device=False):
    """Get serial number of an opened DYNAP-SE1 board.

    Args:
        device_idx (int, optional): DYNAP-SE1 board index. Defaults to 0.
        select_device (bool, optional): whether to select one DYNAP-SE1 board out of 
            multiple connected ones. Defaults to False.

    Returns:
        int: serial_number
    """    
    if type(device_idx) is str:
        warnings.warn("device_name is deprecated, please use device_idx.", DeprecationWarning)
        device_idx = 0
    device_infos = samna.device.get_opened_devices()
    if select_device:
        for i in range(len(device_infos)):
            print("["+str(i)+"]: ", device_infos[i], "serial_number", device_infos[i].serial_number)
        device_idx = input("Select the device you want to open by index: ") 
    else:
        return device_infos[device_idx].serial_number

def print_dynapse1_spike(event):
    """Print the Spike in the following format: "(timestamp, global_neuron_id),".

    :param samna.dynapse1.Spike event: spike.
    """
    print((event.timestamp, event.neuron_id+
            event.core_id*NEURONS_PER_CORE+
            event.chip_id*NEURONS_PER_CHIP), end=',')

def create_neuron_select_graph(model, neuron_ids):
    """
    Create a graph: source_node in model -> filter_node in graph -> sink_node to get events.
    Only filter_node is in the graph. Source and sink nodes are outside graph.
    
    Usage:
    To use the created/returned graph, first graph.start().
    To get events, sink_node.get_events().
    If you graph.stop(), for now the graph actually won't stop, all events are still
    streamed into the buffer of sink_node. This is work in progess.
    Thus to get events for 1 second, you need to first clear the buffer 
    of sink_node using get_events().
    i.e.,
    
    .. code-block::

        sink_node.get_events()
        sleep(1)
        events = sink_node.get_events()

    See details at: https://synsense-sys-int.gitlab.io/samna/filters.html#
    
    Args:
        model (Dynapse1Model): , returned by open_dynapse1()
        neuron_ids (list[tuple(int, int, int)]): tuple in the order of (chip, core, neuron), 
            neuron ids of the neurons you want to monitor.
        
    Returns:
        graph (samna.graph.EventFilterGraph): samna filtering graph.
        filter_node (samna.graph.nodes.Dynapse1NeuronSelect_dynapse1_dynapse1_event): 
            filter_node that filters the events of selected neurons.
        sink_node (samna.BasicSinkNode_dynapse1_dynapse1_event): sink node that receives 
            the events from the filter node.
    """
    # create a graph. A graph is a thread.
    graph = samna.graph.EventFilterGraph()
    
    sink_node = samna.BasicSinkNode_dynapse1_dynapse1_event()

    # NeuronSelectFilterNode. Node 2. Initialized inside graph, by add_filter_node.
    _, filter_node, _ = graph.sequential([model.get_source_node(), "Dynapse1NeuronSelect", sink_node])
    # Get this filterNode from the created graph and set selected neuron IDs.
    filter_node.set_neurons(neuron_ids)

    return graph, filter_node, sink_node

def get_time_wrap_events(model):
    """
    DYNAP-SE1 sends out 2 types of events: spike and timeWrapEvent. 
    timeWrapEvent occurs when the 32 bit timestamp wraps around. 
    This happens every ~37(?) minutes. With the graph created by this method, 
    you can monitor if any timeWrapEvent is generated effectively with the C++ backend.

    Agrs:
        model (samna.dynapse1.Dynapse1Model): returned by open_dynapse1()
    """
    graph = samna.graph.EventFilterGraph()

    sink_node = samna.BasicSinkNode_dynapse1_dynapse1_event()

    _, filter_node, _ = graph.sequential([model.get_source_node(), "Dynapse1TimestampWrapEventFilter", sink_node])

    return graph, sink_node


def set_fpga_spike_gen(fpga_spike_gen, spike_times, indices, target_chips, isi_base, repeat_mode=False, is_first=False):
    """
    Author: Nicoletta Risi. Adapted by Jingyue Zhao.

    This function sets the SpikeGenerator object given a list of spike times, 
    in sec, correspondent input neuron ids and the target chips, i.e. the
    chip destination of each input event. 
    
    About isi_base:
    Given a list of spike times (in sec) a list of isi (Inter Stimulus Interval) 
    is generated. Given a list of isi, the resulting list of isi set from the 
    FPGA will be:

        isi * unit_fpga
        with             
        unit_fpga = isi_base/90 * us    
        
    Thus, given a list of spike_times in sec:
    
        - first the spike times are converted in us
        - then a list of isi (in us) is generated
        - then the list of isi is divided by the unit_fpga (so that the
            resulting list of isi set on FPGA will have the correct unit
            given the input isi_base)

    E.g.: if isi_base=900 the list of generated isi will be multiplied on
    FPGA by 900/90 us = 10 us

    Args:
        fpga_spike_gen (samna.dynapse1.Dynapse1FpgaSpikeGen): Dynapse1FpgaSpikeGen 
            retrieved from Dynapse1Model.
        spike_times    (list): list of input spike times, in second.
        indices     (list): list of FPGA spike generator ids sorted according to 
                            time of spike.
        target_chips    (list): list of target chip to which each event will be
                            sent.
        isi_base        (int): 90 or 900 (See above)
        repeat_mode    (bool): If repeat is True, the spike generator will 
                            loop from the beginning when it reaches the end 
                            of the stimulus.     

    
    """
    assert all(np.sort(spike_times)==(spike_times)), 'Spike times must be sorted!'
    assert(len(indices)==len(spike_times)==len(target_chips)), 'Spike times '+\
        ' and neuron ids need to have the same length'
    
    unit_fpga = isi_base/90 #us
    spike_times_us = np.array(spike_times)*1e6
    spike_times_unit_fpga = (spike_times_us / unit_fpga).astype('int')
    
    fpga_isi = np.array([0]+list(np.diff(spike_times_unit_fpga)), dtype=int)
    fpga_nrn_ids = np.array(indices)
    fpga_target_chips = np.array(target_chips)
    
    fpga_events = []
    for idx_isi, isi in enumerate(fpga_isi):
        fpga_event = dyn1.FpgaSpikeEvent()
        fpga_event.core_mask = 15
        fpga_event.target_chip = fpga_target_chips[idx_isi]
        fpga_event.neuron_id = fpga_nrn_ids[idx_isi]
        fpga_event.isi = isi
        fpga_events.append(fpga_event)
        
    assert all(np.asarray(fpga_isi) < MAX_ISI), 'isi is too large for'+\
            'the specified isi_base!'
    assert len(fpga_isi) < MAX_FPGA_LEN , 'Input stimulus is too long!'
    
    # Set spikeGen:
    if is_first: fpga_spike_gen.set_variable_isi_mode(True)
    fpga_spike_gen.preload_stimulus(fpga_events)
    fpga_spike_gen.set_isi_multiplier(isi_base)
    if is_first: fpga_spike_gen.set_repeat_mode(repeat_mode)

def get_selected_timestamps(spikes, neuron_ids):
    """
    Given Dynapse1Spikes retrieved from NeuronSelect sinknode, 
    filter out timestamps of selected neuron_ids.

    Parameters
    -----------
    spikes : list[Dynapse1Spike]
        result of sink_node.get_events()
    neuron_ids : list[(int,int,int)]
        neuron_ids
    
    Returns
    ---------
    timestamps : list[list[int]]
        spike timestamps of selected neurons. The list indexing is the same as
        the neuron indexing in the list neuron_ids.
    """
    if len(set(neuron_ids)) != len(neuron_ids):
        raise Exception("Duplicate neuron ids exist!")

    num_neurons = len(neuron_ids)
    timestamps = []
    for i in range(num_neurons):
        timestamps.append([])
    
    for spike in spikes:
        spike_neuron = (spike.chip_id, spike.core_id, spike.neuron_id)
        try:
            # if spike_neuron in neuron_ids, add timestamp to corresponding timestamp list
            id_in_list = neuron_ids.index(spike_neuron)
            timestamps[id_in_list].append(spike.timestamp)
        except ValueError:
            # if not, do nothing
            pass
    
    return timestamps

def get_selected_traces(timed_traces, neuron_ids):
    """
    Filter out timestamps and trace_values of selected neuron_ids.

    Parameters
    -----------
    timed_traces : list[Dynapse1Trace]
        result of sink_node.get_events()
    neuron_ids : list[(int,int,int)]
        neuron_ids
    
    Returns
    ---------
    timestamps : list[list[int]]
        spike timestamps of selected neurons. The list indexing is the same as
        the neuron indexing in the list neuron_ids.
    trace_values : list[list[float]]
        trace values at spike times. The list indexing is the same as
        the neuron indexing in the list neuron_ids.
    """
    if len(set(neuron_ids)) != len(neuron_ids):
        raise Exception("Duplicate neuron ids exist!")

    num_neurons = len(neuron_ids)
    timestamps = []
    for i in range(num_neurons):
        timestamps.append([])
    trace_values = []
    for i in range(num_neurons):
        trace_values.append([])
    
    # tracemap over time
    for trace in timed_traces:
        trace_map = trace.trace_map
        timestamp = trace.timestamp

        for i in range(num_neurons):
            neuron_id = neuron_ids[i]
            trace_value = trace_map.get(neuron_id)
            if trace_value != None:
                trace_values[i].append(trace_value)
                timestamps[i].append(timestamp)
    
    return timestamps, trace_values

def get_trace_value(traces, timestamp):
    """
    Get 1 single trace value from a list of Dynapse1Traces.

    Args:
        traces (list[samna.dynapse1.Dynapse1Trace]): list of trace events.
        timestamp (int): DYNAP-SE1 timestamp, in microsecond.

    Returns:
        samna.dynapse1.Dynapse1Trace: the trace with timestamp. If not found, return None.
    """
    for trace in traces:
        if trace.timestamp == timestamp:
            return trace
    
    if timestamp < traces[0].timestamp:
        print("Timestamp%i not found, < list start %i." % (timestamp, traces[0].timestamp))
    if timestamp > traces[-1].timestamp:
        print("Timestamp %i not found, > list end %i." % (timestamp, traces[-1].timestamp))
    return None

def save_samna_objects2file(objects, fname='./spikes.txt'):
    """Save a list of samna objects to json file.

    Args:
        objects (samna objects): list of samna objects, samna.dynapse1.Spike and samna.dynapse1.Dynapse1Trace are tested.
        fname (str, optional): file path and name. Defaults to './spikes.txt'.
    """
    parser_pos = [_.start() for _ in re.finditer('/', fname)]
    path = fname[:parser_pos[-1]]
    if not os.path.exists(path):
        os.makedirs(path)

    list_obj = []
    for obj in objects:
        list_obj.append(json.loads(obj.to_json()))
    
    with open(fname, 'w') as json_file:
        json.dump(list_obj, json_file, indent=4)

def load_samna_objects_file(fname='./spikes.txt'):
    """Load a json file storing samna objects to a list of samna objects.

    Args:
        fname (str, optional): file path and name. Defaults to './spikes.txt'.

    Returns:
        list[samna object]: list of samna objects, samna.dynapse1.Spike and samna.dynapse1.Dynapse1Trace are tested.
    """
    with open(fname) as json_file:
        list_obj_dict = json.load(json_file)
    
    list_obj = []
    for obj_dict in list_obj_dict:
        list_obj.append(convert_dict2samna_object(obj_dict))
    return list_obj

def convert_dict2samna_object(obj_dict):
    """Convert a to_json() string to a samna object.

    Args:
        obj_dict (samna object): samna object, samna.dynapse1.Spike and samna.dynapse1.Dynapse1Trace are supported.

    Raises:
        Exception: if the object is not samna.dynapse1.Spike or samna.dynapse1.Dynapse1Trace, it cannot be parsed by this function.

    Returns:
        samna object: samna.dynapse1.Spike or samna.dynapse1.Dynapse1Trace.
    """
    obj_dict = obj_dict['value0']
    # Dynapse1Trace
    if 'traceMap' in obj_dict.keys():
        trace_event = dyn1.Dynapse1Trace()
        trace_event.timestamp = obj_dict['timestamp']
        trace_map_list = obj_dict['traceMap']
        trace_map = {}
        for trace_item in trace_map_list:
            neuron = (trace_item['key']['tuple_element0'], trace_item['key']
            ['tuple_element1'], trace_item['key']['tuple_element2'])
            trace_value = trace_item['value']
            trace_map.update({neuron:trace_value})
        trace_event.trace_map = trace_map
        trace_event.trigger_neuron = (obj_dict['triggerNeuron']['tuple_element0'],\
            obj_dict['triggerNeuron']['tuple_element1'],obj_dict['triggerNeuron']['tuple_element2'])
        return trace_event
    # spike
    elif 'timestamp' in obj_dict.keys() and 'neuronId' in obj_dict\
    .keys():
        spike = dyn1.Spike()
        spike.timestamp = obj_dict['timestamp']
        spike.neuron_id = obj_dict['neuronId']
        spike.core_id = obj_dict['coreId']
        spike.chip_id = obj_dict['chipId']
        return spike
    else:
        raise Exception("Samna object type not supported by the conversion function.")