import torch
import numpy as np
import yaml
from IFTS.fiber_simulation.utils.show_progress import progress_info
from IFTS.simulation_main.modul_main import sig_main, tx_main, rx_main, channel_main
from IFTS.simulation_main.modul_para import simulation_para, signal_para, txsignal_para, channel_para, rxsignal_para, sigplot_para
 
def tx_part():
    @progress_info(total_num = sig_para.channel_num,\
        infor_print = sig_para.infor_print, discription = ' Tx DSP')
    def main(**kwargs):
        pbar = kwargs.get('pbar', 0)
        i_total = 0
        bit_seq_pol, sym_seq_pol, tx_seq_pol, integer_seq_pol = [], [], [], []
        for i_ch in range(sig_para.channel_num):
            for i_p in range(sig_para.nPol):
                if seed == -1:
                    data_seed = -1
                else:
                    data_seed = seed + i_p*200 + i_ch*501
                bit, sym, sym_map_local, integer = sig_main.sig_tx(sig_para, seed = data_seed)
                bit_seq_pol.append(bit), sym_seq_pol.append(sym), integer_seq_pol.append(integer)
            tx_sam = tx_main.tx(sym_seq_pol, tx_para, plot_para)
            bit_seq.append(bit_seq_pol) 
            sym_seq.append(sym_seq_pol)
            integer_seq.append(integer_seq_pol)
            tx_seq.append(tx_sam)
            sym_map.append(sym_map_local)
            if i_ch == sig_para.cut_idx:
                plot_para.get_colour(integer_seq_pol)
            bit_seq_pol, sym_seq_pol, integer_seq_pol = [], [], []
            i_total += 1
            if type(pbar) != type(0):
                pbar.update(1)
    sym_map = []
    bit_seq, sym_seq, tx_seq, integer_seq = [], [], [], []
    main()
    if tx_para.save_data:
        simu_para.save_data_func(np.array(sym_seq), 'sym_seq', data_mode = 'numpy')
        simu_para.save_data_func(np.array(sym_map), 'sym_map', data_mode = 'numpy')
    return bit_seq, sym_seq, integer_seq, tx_seq, sym_map


seed = 20 # fix random seeds for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if seed != -1:
    torch.manual_seed(seed)
config_path = './config/paras.yml'
with open(config_path, 'r', encoding="utf-8") as f:
    configs = yaml.safe_load(f.read())
if configs['Ch_Para']['fiber_config']['mode'] =='NN' and configs['Simu_Para']['channel_type'] == 1:
    nn_config_path='./config/' + 'paras_'+configs['Ch_Para']['fiber_config']['nn_model']+'.yml'
    with open(nn_config_path, 'r', encoding="utf-8") as f2:
        nn_configs = yaml.safe_load(f2.read())
    for key in nn_configs.keys():
        configs[key].update(nn_configs[key])
    # with open(config_path, "w", encoding="utf-8") as f:
    #     yaml.dump(nn_configs, f,allow_unicode=True)
print('---------------- Parameters initializing... ----------------')
simu_para = simulation_para.Simu_Para(seed, configs)
print(' sig_para initializing...')
sig_para = signal_para.Sig_Para(seed, configs)
print(' tx_para initializing...')
tx_para = txsignal_para.Tx_Para(seed, configs)
print(' rx_para initializing...')
rx_para = rxsignal_para.Rx_para(seed, configs)
print(' ch_para initializing...')
ch_para = channel_para.Ch_Para(seed, configs)
print(' plot_para initializing...')
plot_para = sigplot_para.Plot_Para(seed, configs)
print('---------------- Parameters initialized! -------------------')
if sig_para.infor_print:
    print('---------------- Tx start ----------------')
bit_seq, sym_seq, integer_seq, tx_seq, sym_map = tx_part()
if sig_para.fig_plot:
    plot_para.scatter_plot(sym_seq[sig_para.cut_idx], name = 'Constellations', s=40)
if ch_para.infor_print:
    print('---------------- Channel start ----------------')
rx_sig = channel_main.channel_transmission(tx_seq, ch_para, plot_para = plot_para) 
if rx_para.infor_print:
    print('---------------- Channel finished ----------------')
    print('---------------- Rx DSP started ----------------')
if configs['Simu_Para']['channel_type'] == 2:
    rx_sig_after_dsp = rx_main.rx_awgn(rx_sig, sym_map[rx_para.cut_idx], sym_seq[rx_para.cut_idx], rx_para, plot_para)
else:
    rx_sig_after_dsp = rx_main.rx(rx_sig, sym_map[rx_para.cut_idx], sym_seq[rx_para.cut_idx], rx_para, plot_para)
sig_para, sig_noise, sig_int = sig_main.sig_rx(rx_sig_after_dsp, sym_map[rx_para.cut_idx],\
    bit_seq[rx_para.cut_idx], integer_seq[rx_para.cut_idx], sig_para, plot_para = plot_para)
if sig_para.fig_plot:
    plot_para.constellation_points = sig_noise[0].shape[0]
    plot_para.front_sym_num = 0
    plot_para.get_colour(sig_int)
    plot_para.scatter_plot_nPol(sig_noise, sam_num = 1, name = 'Noise_After_DSP', set_c = 0)
    plot_para.psd_nPol(sig_noise, name = 'Noise_PSD_After_DSP')
if rx_para.infor_print:
    print('---------------- Rx DSP finished ----------------')
    
results = open(simu_para.result_path + 'results.txt','w+')
simu_infor = open(simu_para.result_path + 'simu_para.txt','w+')
sig_infor = open(simu_para.result_path + 'sig_para.txt','w+')
tx_infor = open(simu_para.result_path + 'tx_para.txt','w+')
ch_infor = open(simu_para.result_path + 'ch_para.txt','w+')
rx_infor = open(simu_para.result_path + 'rx_para.txt','w+')
simu_para.print_para(simu_infor)
sig_para.print_para(sig_infor)
tx_para.print_para(tx_infor)
rx_para.print_para(rx_infor)
if sig_para.infor_print:
    print('Polarization X, BER: {:.6f} B2Q: {:.6f} C2Q: {:.6f}'.format(\
        sig_para.ber_array[0], sig_para.b2q_array[0], sig_para.c2q_array[0]), file = results)
    print('Polarization Y, BER: {:.6f} B2Q: {:.6f} C2Q: {:.6f}'.format(\
        sig_para.ber_array[1], sig_para.b2q_array[1], sig_para.c2q_array[1]), file = results)
    print('GMI: {:.6f} MI: {:.6f}'.format(sig_para.gmi_value, sig_para.mi_value), file = results)
results.close()
    