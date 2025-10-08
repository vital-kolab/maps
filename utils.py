import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt


def get_percent_correct_from_proba(prob, labels,class_order, eps=1e-3):
    nrImages = prob.shape[0]
    
    _,ind = np.unique(labels, return_index=True)
    class_order=labels[np.sort(ind)] #np.unique(labels)
    pc = np.zeros((nrImages,len(class_order)))
    pc[:]=np.NAN
    
    for i in range(nrImages):
        loc_target = labels[i]==class_order
        pc[i,:] = np.divide(prob[i,labels[i]==class_order]+eps,(prob[i,:]+eps)+(prob[i,loc_target]+eps)) #+eps
        pc[i,loc_target]=np.NAN
    return pc

def create_i1_test(new_features, nrImages=200, eps=1e-5):
    lb = ["bear", "elephant", "person", "car", "dog", "apple", "chair", "plane", "bird", "zebra"]
    labels = np.repeat(lb, nrImages // len(lb), axis=0)

    i1 = np.zeros((nrImages), dtype=float)
    i1[:] = np.NAN

    pc = get_percent_correct_from_proba(new_features, labels, np.array(lb), eps)
    i1[:] = np.nanmean(pc, axis=1)

    return i1

def spearman_ci_boot(x, y, n_boot=5000, ci=95, rng=None):
    x, y = np.asarray(x), np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if rng is None: rng = np.random.default_rng()

    
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = spearmanr(x[idx], y[idx]).correlation

    r0 = np.nanmean(boots)

    lo = np.percentile(boots, (100-ci)/2)
    hi = np.percentile(boots, 100-(100-ci)/2)
    # use half-width for error bars
    err_low  = r0 - lo
    err_high = hi - r0
    return r0, err_low, err_high

def get_gt_single(model1, model2, ground_truth, method_names):
    gt_ref = np.zeros((12, 200))
    model_pair_1 = (model1, model2)
    model_pair_2 = (model2, model1)

    # print(model_pair_1, model_pair_2)

    # Check which ordering exists in ground_truth and use it
    pair_ground_truth = {
        method_name: ground_truth[method_name].get(model_pair_1, ground_truth[method_name].get(model_pair_2, np.nan))
        for method_name in method_names
    }

    for i2, m in enumerate(method_names):
        gt_ref[i2] = pair_ground_truth[m]
    return gt_ref

def spearmanbrown_correction(var):  # Spearman Brown Correct the correlation value
    spc_var = (2 * var) / (1 + var)
    return spc_var

def countmin(data, key):
    unique_images = data[key].unique()
    num_trials = [sum(data[key] == img) for img in unique_images]
    return min(num_trials)

def calculate_i1(data, key):
    rng = np.random.default_rng()
    
    n = countmin(data, key) # min. number of trials per image
    uImgs = np.sort(data[key].unique())
    
    i1 = np.full(len(uImgs), np.nan)

    for i, img in enumerate(uImgs):
        loc = np.where(data[key] == img)[0]

        sampled_locs = rng.choice(loc, n, replace=False)
        #sampled_locs = loc[:n]
        i1[i] = np.nanmean(data.iloc[sampled_locs]['correct'])

    return i1

# Function to calculate i1_values for a specific trial split
def calculate_i1_per_trial(data, trial_indices, key):
    uImgs = data[key].unique()
    i1 = np.full(len(uImgs), np.nan)
    
    for i, img in enumerate(uImgs):
        loc = np.where(data[key] == img)[0]
        split_loc = np.intersect1d(loc, trial_indices)
        if len(split_loc) > 0:
            i1[i] = np.nanmean(data.loc[split_loc, 'correct'])
    return i1

# Function to compute split-half reliability
def split_half_reliability(data, key):
    n = countmin(data, key)  # Minimum number of trials per image
    uImgs = np.sort(data[key].unique())
    rng = np.random.default_rng()  # Random number generator

    # Split trials for each image into two halves
    all_split1, all_split2 = [], []
    for img in uImgs:
        loc = np.where(data[key] == img)[0]
        if len(loc) >= n:  # Ensure there are enough trials
            sampled_locs = rng.choice(loc, n, replace=False)
            split1, split2 = np.array_split(sampled_locs, 2)
            all_split1.extend(split1)
            all_split2.extend(split2)

    # Calculate i1 values for each split
    i1_split1 = calculate_i1_per_trial(data, all_split1, key)
    i1_split2 = calculate_i1_per_trial(data, all_split2, key)

    # Compute correlation between splits
    valid_indices = ~np.isnan(i1_split1) & ~np.isnan(i1_split2)
    correlation, _ = pearsonr(i1_split1[valid_indices], i1_split2[valid_indices])
    correlation_corrected = spearmanbrown_correction(correlation)

    return correlation_corrected, i1_split1, i1_split2

def calculate_responses_per_trial_clean(data, trial_indices, key="image_num"):
    uImgs = data[key].unique()
    i1 = np.full(len(uImgs), np.nan)

    img_array = data[key].to_numpy()
    correct_pos = data.columns.get_loc('correct')
    
    for i, img in enumerate(uImgs):
        loc = np.flatnonzero(img_array == img)                 # positional indices for this image
        split_loc = np.intersect1d(loc, trial_indices)         # still positional
        if len(split_loc) > 0:
            i1[i] = np.nanmean(data.iloc[split_loc, correct_pos])
    return i1

def split_half_reliability_clean(data, key="image_num"):
    n = countmin(data, key)  # Minimum number of trials per image
    # print(n)
    uImgs = np.sort(data[key].unique())
    # print(uImgs)
    rng = np.random.default_rng()  # Random number generator

    all_split1, all_split2 = [], []
    img_array = data[key].to_numpy()

    # Split trials for each image into two halves
    for img in uImgs:
        loc = np.flatnonzero(img_array == img)                 # positional indices for this image
        if len(loc) >= n:
            sampled = rng.choice(loc, n, replace=False)
            split1, split2 = np.array_split(sampled, 2)        # equal since n is even
            all_split1.extend(split1.tolist())
            all_split2.extend(split2.tolist())

    i1_split1 = calculate_responses_per_trial_clean(data, np.array(all_split1, dtype=int), key)
    i1_split2 = calculate_responses_per_trial_clean(data, np.array(all_split2, dtype=int), key)

    valid = ~np.isnan(i1_split1) & ~np.isnan(i1_split2)
    if valid.sum() < 3:
        return np.nan  # too few images to correlate

    r, _ = pearsonr(i1_split1[valid], i1_split2[valid])
    return spearmanbrown_correction(r)

def journal_figure(do_save=False, filename='figure.eps', dpi=300, size_inches=(2.16, 2.16), linewidth=1):
    """
    Adjusts the current matplotlib figure to make it look publication-worthy.
    
    Parameters:
    - do_save: bool, whether to save the figure to an EPS file.
    - filename: str, the name of the file to save the figure as.
    - dpi: int, the resolution of the figure in dots per inch.
    - size_inches: tuple, the size of the figure in inches.
    - linewidth: float, the line width for the plot elements.
    """
    ax = plt.gca()  # Get the current axes
    
    # Adjust tick direction and length
    ax.tick_params(direction='out', length=10, width=linewidth)
    
    # Turn off the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=6, width=2)
    ax.set_aspect(1.0/plt.gca().get_data_ratio(), adjustable='box')
    # Set font size and type
    plt.xticks(fontsize=12) #, fontname='Times New Roman')
    plt.yticks(fontsize=12) #, fontname='Times New Roman')
    
    if do_save:
        # Save the figure
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='eps', linewidth=linewidth)