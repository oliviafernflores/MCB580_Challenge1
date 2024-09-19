from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def Direct(y, t, TF1, k1, k2, Kd1, total_time):
    if t > total_time / 3:
        TF1 = 0
    if t > total_time * 2/3:
        TF1 = 1
    d_dt = np.zeros(1)
    d_dt[0] = k1*(1 - 1/(1+ (TF1/Kd1))) - k2*y[0]
    return d_dt

def integrate_Direct(y0, time_steps, params):
    total_time = time_steps[-1]
    result = odeint(
        lambda y, t: Direct(y, t, params['TF1'], params['k1'], params['k2'], params['Kd1'], total_time),
        y0, time_steps
    )
    return result

def parameter_scan(param_change, args, y0, time_steps):
    param_vals = np.logspace(-2, 2, 5)
    num_vals = len(param_vals)
    args_array = [{} for _ in range(num_vals)]
    keys = list(args.keys())
    if param_change in keys:
        param_index = keys.index(param_change)
        for j in range(num_vals):
            updated_args = args.copy()
            updated_args[param_change] = param_vals[j]
            args_array[j] = updated_args
    return args_array

def calc_plot_scan_results(name, arr, y0, time_steps, time):
    results = []
    labels = []
    for params in arr:
        param_values = {param: params[param] for param in ['TF1', 'k1', 'k2', 'Kd1']}
        result = integrate_Direct(y0, time_steps, param_values)
        results.append(result)
        label = f'{name}={params[name]}' if name in params else 'Unknown parameter'
        labels.append(label)
    plt.figure()
    for result, label in zip(results, labels):
        plt.plot(time_steps, result[:, 0], label=label)    
    plt.xlabel('Time')
    plt.ylabel('[mRNA$_1$]')
    plt.legend(loc='best')
    plt.title(f'{name} Parameter Scan Results')
    plt.savefig('DirectResults/Direct_' + str(name) + '_' + str(time) + '_scan_results.pdf')
    
def do_param_scans(params, y0, time_steps, time):
    for key in params.keys():
        if key == 'TF1':
            continue
        else:
            array = parameter_scan(key, params, y0, time_steps)
            calc_plot_scan_results(key, array, y0, time_steps, time)
            
def main():
    params = {'TF1':1, 'k1':1, 'k2':1, 'Kd1':1}
    y0 = [0]
    
    time_steps = np.linspace(0, 100, 1000)
    do_param_scans(params, y0, time_steps, 100)
    
    time_steps = np.linspace(0, 1000, 10000)
    do_param_scans(params, y0, time_steps, 1000)
    
    time_steps = np.linspace(0, 10000, 10000)
    do_param_scans(params, y0, time_steps, 10000)

    time_steps = np.linspace(0, 100000, 100000)
    do_param_scans(params, y0, time_steps, 100000)
    
    time_steps = np.linspace(0, 1000000, 1000000)
    do_param_scans(params, y0, time_steps, 1000000)

if __name__ == '__main__':
    main()
