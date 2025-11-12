import torch
import numpy as np
import yaml
import os
from IFTS.fiber_simulation.utils.show_progress import progress_info
from IFTS.simulation_main.modul_main import sig_main, tx_main, rx_main, channel_main
from IFTS.simulation_main.modul_para import simulation_para, signal_para, txsignal_para, channel_para, rxsignal_para, sigplot_para

# ========== 新增信号提取功能 ==========
def extract_input_signals(bit_seq, sym_seq, tx_seq, integer_seq, sym_map, sig_para, tx_para, save_path='./results/input_signals/'):
    """提取所有输入信号数据"""
    os.makedirs(save_path, exist_ok=True)
    print("=== 提取输入信号数据 ===")
    
    input_data = {}
    
    # 1. 提取比特序列
    print("提取比特序列...")
    bit_data = []
    for i_ch, channel_bits in enumerate(bit_seq):
        channel_bit_data = []
        for i_pol, pol_bits in enumerate(channel_bits):
            if torch.is_tensor(pol_bits):
                bits = pol_bits.cpu().numpy() if pol_bits.is_cuda else pol_bits.numpy()
            else:
                bits = np.array(pol_bits)
            channel_bit_data.append(bits)
        bit_data.append(channel_bit_data)
    
    # 2. 提取符号序列（星座图数据）
    print("提取符号序列...")
    symbol_data = []
    constellation_data = []
    for i_ch, channel_symbols in enumerate(sym_seq):
        channel_symbol_data = []
        for i_pol, pol_symbols in enumerate(channel_symbols):
            if torch.is_tensor(pol_symbols):
                symbols = pol_symbols.cpu().numpy() if pol_symbols.is_cuda else pol_symbols.numpy()
            else:
                symbols = np.array(pol_symbols)
            
            # 转换为复数形式
            if symbols.ndim > 1 and symbols.shape[-1] == 2:
                complex_symbols = symbols[..., 0] + 1j * symbols[..., 1]
            else:
                complex_symbols = symbols
            
            channel_symbol_data.append(complex_symbols)
            constellation_data.append(complex_symbols)  # 用于星座图
        symbol_data.append(channel_symbol_data)
    
    # 3. 提取发射波形
    print("提取发射波形...")
    tx_waveform_data = []
    for i_ch, channel_tx in enumerate(tx_seq):
        if torch.is_tensor(channel_tx):
            tx_signal = channel_tx.cpu().numpy() if channel_tx.is_cuda else channel_tx.numpy()
        else:
            tx_signal = np.array(channel_tx)
        tx_waveform_data.append(tx_signal)
    
    # 4. 提取整数序列（编码数据）
    print("提取整数编码序列...")
    integer_data = []
    for i_ch, channel_ints in enumerate(integer_seq):
        channel_int_data = []
        for i_pol, pol_ints in enumerate(channel_ints):
            if torch.is_tensor(pol_ints):
                integers = pol_ints.cpu().numpy() if pol_ints.is_cuda else pol_ints.numpy()
            else:
                integers = np.array(pol_ints)
            channel_int_data.append(integers)
        integer_data.append(channel_int_data)
    
    # 5. 保存所有数据
    np.savez(f"{save_path}input_signals.npz",
             bit_sequences=bit_data,
             symbol_sequences=symbol_data,
             tx_waveforms=tx_waveform_data,
             integer_sequences=integer_data,
             constellation_map=sym_map)
    
    # 6. 保存文本摘要
    with open(f"{save_path}input_signals_summary.txt", 'w') as f:
        f.write("输入信号数据摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"信道数量: {len(bit_data)}\n")
        f.write(f"偏振数量: {sig_para.nPol}\n")
        f.write(f"调制方式: {getattr(sig_para, 'modulation', '未知')}\n")
        f.write(f"符号速率: {getattr(sig_para, 'symbol_rate', '未知')}\n")
        f.write(f"采样速率: {getattr(sig_para, 'sample_rate', '未知')}\n")
        f.write("\n各信道符号形状:\n")
        for i, symbols in enumerate(symbol_data):
            f.write(f"  信道{i}: {symbols[0].shape}\n")
    
    print(f"输入信号数据已保存到: {save_path}")
    return bit_data, symbol_data, tx_waveform_data, integer_data

def extract_output_signals(rx_sig, rx_sig_after_dsp, sig_noise, sig_int, sig_para, rx_para, save_path='./results/output_signals/'):
    """提取所有输出信号数据"""
    os.makedirs(save_path, exist_ok=True)
    print("=== 提取输出信号数据 ===")
    
    # 1. 提取原始接收波形
    print("提取原始接收波形...")
    if torch.is_tensor(rx_sig):
        raw_rx_waveform = rx_sig.cpu().numpy() if rx_sig.is_cuda else rx_sig.numpy()
    else:
        raw_rx_waveform = np.array(rx_sig)
    
    # 2. 提取DSP后接收波形
    print("提取DSP后接收波形...")
    if torch.is_tensor(rx_sig_after_dsp):
        processed_rx_waveform = rx_sig_after_dsp.cpu().numpy() if rx_sig_after_dsp.is_cuda else rx_sig_after_dsp.numpy()
    else:
        processed_rx_waveform = np.array(rx_sig_after_dsp)
    
    # 3. 提取噪声信号
    print("提取噪声信号...")
    noise_data = []
    if isinstance(sig_noise, (list, tuple)):
        for noise in sig_noise:
            if torch.is_tensor(noise):
                noise_array = noise.cpu().numpy() if noise.is_cuda else noise.numpy()
            else:
                noise_array = np.array(noise)
            noise_data.append(noise_array)
    else:
        if torch.is_tensor(sig_noise):
            noise_array = sig_noise.cpu().numpy() if sig_noise.is_cuda else sig_noise.numpy()
        else:
            noise_array = np.array(sig_noise)
        noise_data.append(noise_array)
    
    # 4. 提取解调后的编码数据
    print("提取解调编码数据...")
    if torch.is_tensor(sig_int):
        demodulated_data = sig_int.cpu().numpy() if sig_int.is_cuda else sig_int.numpy()
    else:
        demodulated_data = np.array(sig_int)
    
    # 5. 保存所有数据
    np.savez(f"{save_path}output_signals.npz",
             raw_rx_waveform=raw_rx_waveform,
             processed_rx_waveform=processed_rx_waveform,
             noise_signals=noise_data,
             demodulated_data=demodulated_data)
    
    # 6. 保存性能指标
    with open(f"{save_path}output_signals_summary.txt", 'w') as f:
        f.write("输出信号数据摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"原始接收信号形状: {raw_rx_waveform.shape}\n")
        f.write(f"处理后的信号形状: {processed_rx_waveform.shape}\n")
        f.write(f"解调数据形状: {demodulated_data.shape}\n")
        f.write(f"偏振X BER: {sig_para.ber_array[0]:.6f}\n")
        f.write(f"偏振Y BER: {sig_para.ber_array[1]:.6f}\n")
        f.write(f"GMI: {sig_para.gmi_value:.6f}\n")
        f.write(f"MI: {sig_para.mi_value:.6f}\n")
    
    print(f"输出信号数据已保存到: {save_path}")
    return raw_rx_waveform, processed_rx_waveform, noise_data, demodulated_data

def save_constellation_plots(symbol_data, sig_para, save_path='./results/constellations/'):
    """保存星座图"""
    os.makedirs(save_path, exist_ok=True)
    print("生成星座图...")
    
    import matplotlib.pyplot as plt
    
    for i_ch, channel_symbols in enumerate(symbol_data):
        for i_pol, symbols in enumerate(channel_symbols):
            plt.figure(figsize=(8, 8))
            plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.6, s=10)
            plt.title(f'星座图 - 信道{i_ch} 偏振{i_pol}')
            plt.xlabel('同相分量 (I)')
            plt.ylabel('正交分量 (Q)')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.savefig(f'{save_path}constellation_ch{i_ch}_pol{i_pol}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"星座图已保存到: {save_path}")

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

# ========== 新增：提取输入信号 ==========
print('\n' + '='*60)
bit_data, symbol_data, tx_waveform_data, integer_data = extract_input_signals(
    bit_seq, sym_seq, tx_seq, integer_seq, sym_map, sig_para, tx_para
)

# ========== 新增：保存星座图 ==========
save_constellation_plots(symbol_data, sig_para)

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

# ========== 新增：提取输出信号 ==========
print('\n' + '='*60)
raw_rx, processed_rx, noise_data, demodulated_data = extract_output_signals(
    rx_sig, rx_sig_after_dsp, sig_noise, sig_int, sig_para, rx_para
)

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

print('\n' + '='*60)
print('所有信号数据提取完成！')
print('生成的文件:')
print('  - results/input_signals/input_signals.npz (输入信号数据)')
print('  - results/input_signals/input_signals_summary.txt (输入信号摘要)')
print('  - results/output_signals/output_signals.npz (输出信号数据)')
print('  - results/output_signals/output_signals_summary.txt (输出信号摘要)')
print('  - results/constellations/ (星座图PNG文件)')
print('='*60)