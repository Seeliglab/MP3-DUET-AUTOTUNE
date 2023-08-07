#AML
#reduce the DUET outputs to have true orthogonality gaps (assuming non-measured interactions are not interactions)
import pandas as pd 
import numpy as np
import matplotlib as mpl
from multiprocessing import Pool
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import double
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from graph_tool.all import *

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

color_dict = {'HT_DHD': '#8787def7', 'ZCON': '#d35f5ff7', 'DHDr2':'#ff9955f7', 'MALB': '#ffe680f7', 'OTHER':'red'}
# vprop.


def number_nans(binders_set, df, col):
    square_df = np.empty((len(binders_set), len(binders_set)))
    square_df[:] = np.nan
    for i in range(0, len(binders_set)):
        for j in range(0, i+1):# in bettter_order_binders:
                ppi_id = [binders_set[i], binders_set[j]]
                ppi_id.sort()
                if ':'.join(ppi_id) in df.PPI.to_list():
                    if df[df.PPI == ':'.join(ppi_id)][col].to_numpy():
                        square_df[i,j] = df[df.PPI == ':'.join(ppi_id)][col].to_numpy()
    return sum(np.arange(1,len(binders_set)+1) - np.sum(~np.isnan(square_df), axis = 1))

def random_draw_nan_count(subset_pros, df_frame):
    return number_nans(subset_pros, df_frame, 'ashr_log2FoldChange_HIS_TRP')

def draw_num_nans_perm_test(df, n_draws, all_pros, pro_set):
    pro_set_nan_count = number_nans(pro_set, df, 'ashr_log2FoldChange_HIS_TRP')
    draws = []
    for _ in range(0, n_draws):
        draws.append(list(np.random.choice(all_pros, len(pro_set), replace=False)))
    with Pool() as pool:
        results = pool.map(partial(random_draw_nan_count, df_frame=df), draws)
    return pro_set_nan_count, results


def mark_type_pro(name_pro):
    if 'HT_DHD' in name_pro:# and 'HT_DHD' in name2 and name1 != name2:
        return 'HT_DHD'
    elif 'ZCON' in name_pro:# and 'ZCON' in name2 and name1 != name2:
         return 'ZCON'
    elif 'DHDr2' in name_pro:# and 'ZCON' in name2 and name1 != name2:
         return 'DHDr2'
    return 'MALB'


def mark_targets(name1, name2):
    if 'HT_DHD' in name1 and 'HT_DHD' in name2 and name1 != name2:
        if  name1[2:] == name2[2:]:
            return True
    elif 'ZCON' in name1 and 'ZCON' in name2 and name1 != name2:
        if  name1[2:] == name2[2:]:
            return True
    elif 'DHDr2' in name1 and 'DHDr2' in name2 and name1 != name2:
        if name1[:-2] == name2[:-2]:
         #   print("        YES")
            return True
    return False

def make_graphviz_graph_fixed_location(scanned_sol, locs, output_name):
    print (scanned_sol)
    scanned_sol_pros = list(set(scanned_sol.pro1.to_list() + scanned_sol.pro2.to_list()))
    g = Graph(directed=False)
    graph_inds_dict = {}

    print (len(scanned_sol_pros), len(set(scanned_sol_pros)))
    for b in scanned_sol_pros:
        graph_inds_dict[b] = [scanned_sol_pros.index(b)]

    scanned_sol['node1'] = scanned_sol.pro1.apply(lambda x: graph_inds_dict[x][0])
    scanned_sol['node2'] = scanned_sol.pro2.apply(lambda x: graph_inds_dict[x][0])
    #print (graph_inds_dict)
    eprop = g.new_edge_property('double')
    for ind, row in scanned_sol.iterrows():
        curr_edge = g.add_edge(row['node1'], row['node2'])
        eprop[curr_edge] = float(row['ashr_log2FoldChange_HIS_TRP'])

    #see how many are on target vs not
    color_prop = g.new_vertex_property("string")
    position_v = g.new_vertex_property("vector<double>")
    for b in graph_inds_dict:
        color_prop[graph_inds_dict[b][0]] = color_dict[mark_type_pro(b)]
        position_v[graph_inds_dict[b][0]] = (float(locs[locs.Binder == b].x)*20, float(locs[locs.Binder == b].y)*20)
    g.vertex_properties["color"]=color_prop
    g.edge_properties["ashr"] = eprop
    g.vertex_properties["position"]=position_v

    verts_degrees = []
    index_verts = []
    for v in g.vertices():
        verts_degrees.append(v.out_degree())
    print(pd.DataFrame({'dummy': verts_degrees}).dummy.value_counts())

    graph_draw(
        g,
        vertex_pen_width=0.5,
        pos=g.vertex_properties["position"],
        vertex_shape="circle",
        vertex_fill_color=g.vertex_properties["color"],
        edge_pen_width=g.edge_properties["ashr"],
        output = output_name
    )

def get_jerala_target_ppis():
    target_ppis = []
    for i in range(1,13,2):
        ppi_label = ['Jerala_P' + str(i) , 'Jerala_P' + str(i+1) ]
        ppi_label.sort()
        target_ppis.append(':'.join(ppi_label))
    return target_ppis




#duet results DHD0 & DHD1
#design_screening/duet_dhd0_dhd1_results.csv
def get_orthogonality_gapped_duet_results(duet_result_df, all_measurements_df, bad_pros, stopping_size):
    duet_result_df['duet'] = True
    all_measurements_df['pro1'] = all_measurements_df.PPI.apply(lambda x: x.split(':')[0])
    all_measurements_df['pro2'] = all_measurements_df.PPI.apply(lambda x: x.split(':')[1]) 
    subset_pros_l44_l49 = list(set(duet_result_df.pro1.to_list() + duet_result_df.pro2.to_list()))
    subset = all_measurements_df[(all_measurements_df.pro1.isin(subset_pros_l44_l49)) & (all_measurements_df.pro2.isin(subset_pros_l44_l49)) & (all_measurements_df.ashr_log2FoldChange_HIS_TRP > 0)]
    subset = subset.merge(duet_result_df[['PPI', 'duet']], on = 'PPI', how = 'left')
    subset['duet'] = subset['duet'].fillna(value=False)
    subset = subset[~(subset.pro1.isin(bad_pros)) & ~(subset.pro2.isin(bad_pros)) ]

    diminuendo = subset[(subset.duet == True) & (subset.ashr_log2FoldChange_HIS_TRP >= subset[subset.duet == False].ashr_log2FoldChange_HIS_TRP.max())]
    diminuendo_pros = list(set(diminuendo.pro1.to_list() + diminuendo.pro2.to_list()))

    ordered_less_bad = []
    #largest.sort_values('ashr_log2FoldChange_HIS_TRP')
    for ind, row in diminuendo.sort_values('ashr_log2FoldChange_HIS_TRP').iterrows():
        ordered_less_bad.append(row['pro1'])
        ordered_less_bad.append(row['pro2'])
        
    only_duet = all_measurements_df[(all_measurements_df.pro1.isin(diminuendo_pros)) & (all_measurements_df.pro2.isin(diminuendo_pros)) & (all_measurements_df.ashr_log2FoldChange_HIS_TRP > 0)]
    subset2 = only_duet.merge(diminuendo[['PPI', 'duet']], on = 'PPI', how = 'left')
    subset2['duet'] = subset2['duet'].fillna(value=False)
    good_deletion_order = pd.DataFrame({'pro1':ordered_less_bad, 'order':list(range(len(ordered_less_bad)))})
    subset2 = subset2.merge(good_deletion_order, on ='pro1')
    good_deletion_order = pd.DataFrame({'pro2':ordered_less_bad, 'order':list(range(len(ordered_less_bad)))})
    subset2 = subset2.merge(good_deletion_order, on = 'pro2', suffixes = ['_1', '_2'])
    subset2['del_target'] = subset2.apply(lambda row: min(row['order_1'], row['order_2']), axis = 1)

    list_removed = []
    size_set_l39 = [subset2[subset2.duet].shape[0]]
    gap_l39 = [subset2[subset2.duet == True].ashr_log2FoldChange_HIS_TRP.min() - subset2[subset2.duet == False].ashr_log2FoldChange_HIS_TRP.max()]
    best_pros = None
    for ind, row in subset2[subset2.duet == False].sort_values('ashr_log2FoldChange_HIS_TRP', ascending = False).iterrows():
        if row['del_target']%2 == 0:
            #get the index after it also 
            partner_ind = row['del_target'] + 1
        else:
            partner_ind = row['del_target'] - 1
        #print (row['del_target'],partner_ind)
        list_removed += [ordered_less_bad[int(row['del_target'])], ordered_less_bad[int(partner_ind)]]
        #check current gap & size 
        remaining = subset2[~(subset2.pro1.isin(list_removed)) & ~(subset2.pro2.isin(list_removed))]
        #print (remaining.duet.value_counts(), remaining[remaining.duet == True].ashr_log2FoldChange_HIS_TRP.min() - remaining[remaining.duet == False].ashr_log2FoldChange_HIS_TRP.max())
        size_set_l39.append(remaining[remaining.duet].shape[0])
        gap_l39.append(remaining[remaining.duet == True].ashr_log2FoldChange_HIS_TRP.min() - remaining[remaining.duet == False].ashr_log2FoldChange_HIS_TRP.max())
        if remaining[remaining.duet].shape[0] == 2:
            break
        if remaining[remaining.duet].shape[0] == stopping_size:
            #print(list(set(remaining[remaining.duet].pro1.to_list() + remaining[remaining.duet].pro2.to_list())))
            best_pros = list(set(remaining[remaining.duet].pro1.to_list() + remaining[remaining.duet].pro2.to_list()))
            l39_special_gap = gap_l39[-1]
    return gap_l39, size_set_l39, best_pros, l39_special_gap

def main():
    #getting orthogonality gaps of other benchmark sets 
    jerala_target_ppis = get_jerala_target_ppis()
    jerala_homo = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_psuedoreplicate_autotune.csv')
    jerala_homo = jerala_homo.rename(columns = {'Unnamed: 0': 'PPI'})
    jerala_homo['on_target'] = jerala_homo.PPI.isin(jerala_target_ppis)
    jerala_diff = jerala_homo[jerala_homo.on_target == True].ashr_log2FoldChange_HIS_TRP.min() - jerala_homo[jerala_homo.on_target == False].ashr_log2FoldChange_HIS_TRP.max()

    print (jerala_diff)

    #bcl values 
    on_target_interactions = [
                            ['Bcl-2','alphaBCL2'],
                            ['Mcl1[151-321]','alphaMCL1'],
                            ['Bfl-1','alphaBFL1'],
                            ['Bcl-B','alphaBCLB']
                            ]
    bcls = ['Bcl-2', 'Mcl1[151-321]','Bcl-B', 'Bfl-1']
    binders = ['alphaBCL2', 'alphaMCL1', 'alphaBCLB','alphaBFL1']
    on_target_ppis = []
    for ot in on_target_interactions:
        ot.sort()
        on_target_ppis.append(':'.join(ot))

    def mark_type(x):
        if x in on_target_ppis:
            return True
        return False

    homo_m2= pd.read_csv('../processing_pipeline/merged_replicates/deseq_bcl_psuedoreplicate_autotune.csv')
    homo_m2 = homo_m2.rename(columns = {'Unnamed: 0': 'PPI'})
    homo_m2['pro1'] = homo_m2.PPI.apply(lambda x: x.split(':')[0])
    homo_m2['pro2'] = homo_m2.PPI.apply(lambda x: x.split(':')[1]) 
    homo_m2 = homo_m2[((homo_m2.pro1.isin(bcls)) & (homo_m2.pro2.isin(binders))) | ((homo_m2.pro1.isin(binders)) & (homo_m2.pro2.isin(bcls)))]
    homo_m2['type'] = homo_m2.PPI.apply(lambda x: mark_type(x))
    bcl_diff = homo_m2[homo_m2.type == True].ashr_log2FoldChange_HIS_TRP.min() - homo_m2[homo_m2.type == False].ashr_log2FoldChange_HIS_TRP.max()


    duet_results = pd.read_csv('duet_dhd0_dhd1_results.csv')
    duet_pros = list(set(duet_results.pro1.to_list() + duet_results.pro2.to_list()))
    print (duet_results.shape[0])
    full_largest = pd.read_csv('../processing_pipeline/merged_replicates/deseq_dhd1_dhd0_psuedoreplicate_autotune.csv')
    full_largest = full_largest.rename(columns = {'Unnamed: 0': 'PPI'})
    full_largest['pro1'] = full_largest.PPI.apply(lambda x: x.split(':')[0])
    full_largest['pro2'] = full_largest.PPI.apply(lambda x: x.split(':')[1])
    all_pros = list(set(full_largest.pro1.to_list() + full_largest.pro2.to_list()))
    locations = pd.read_csv('draw_l39.csv')
    gap, size_set, best_pros, special_gap = get_orthogonality_gapped_duet_results(duet_results, full_largest, ['A_HT_DHD_73'], 9)
    #filter duet values and draw 
    duet_results_1 = full_largest[full_largest.pro1.isin(best_pros) & full_largest.pro2.isin(best_pros)]
    print (duet_results)
    #make_graphviz_graph_fixed_location(duet_results_1, locations, 'orthogonal_duet_dhd0_dhd1.svg')
    print (duet_results.columns)
    n_samples = 1000
    val_comp, all_vals = draw_num_nans_perm_test(full_largest, n_samples, all_pros, duet_pros)
    #fig, ax = plt.subplots(1, figsize=(2,1))
    boxes = sns.displot(all_vals, height=2, aspect=2)
    plt.axvline(val_comp)
    plt.savefig('orthogonal_duet_dhd0_dhd1_random_nans.svg')
    plt.close()
    #plt.close()



    duet_results = pd.read_csv('duet_dhd2_dhd0_malb_results.csv')
    duet_pros = list(set(duet_results.pro1.to_list() + duet_results.pro2.to_list()))
    full_largest = pd.read_csv('../processing_pipeline/merged_replicates/deseq_dhd2_dhd0_malb_psuedoreplicate_autotune.csv')
    full_largest = full_largest.rename(columns = {'Unnamed: 0': 'PPI'})
    full_largest['pro1'] = full_largest.PPI.apply(lambda x: x.split(':')[0])
    full_largest['pro2'] = full_largest.PPI.apply(lambda x: x.split(':')[1])
    all_pros = list(set(full_largest.pro1.to_list() + full_largest.pro2.to_list()))
    gap_2, size_set_2, best_pros_2, special_gap_2 = get_orthogonality_gapped_duet_results(duet_results, full_largest, [], 6)
    #filter duet values and draw 
    locations = pd.read_csv('draw_l43.csv')
    duet_results_2 = full_largest[full_largest.pro1.isin(best_pros_2) & full_largest.pro2.isin(best_pros_2)]
    #make_graphviz_graph_fixed_location(duet_results_2, locations, 'orthogonal_duet_dhd0_dhd2_malb.svg')
    val_comp, all_vals = draw_num_nans_perm_test(full_largest, n_samples, all_pros, duet_pros)
    boxes = sns.displot(all_vals, height=2, aspect=2)
    plt.axvline(val_comp)
    plt.savefig('orthogonal_duet_dhd0_dhd2_malb_random_nans.svg')
    plt.close()
    #plt.close()

    duet_results = pd.read_csv('duet_all_designed_coils_malb_results.csv')
    duet_pros = list(set(duet_results.pro1.to_list() + duet_results.pro2.to_list()))
    full_largest = pd.read_csv('../processing_pipeline/merged_replicates/deseq_all_designed_coils_psuedoreplicate_autotune.csv')
    full_largest = full_largest.rename(columns = {'Unnamed: 0': 'PPI'})
    full_largest['pro1'] = full_largest.PPI.apply(lambda x: x.split(':')[0])
    full_largest['pro2'] = full_largest.PPI.apply(lambda x: x.split(':')[1])
    all_pros = list(set(full_largest.pro1.to_list() + full_largest.pro2.to_list()))
    gap_3, size_set_3, best_pros_3, special_gap_3 = get_orthogonality_gapped_duet_results(duet_results, full_largest, [], 16)
    #filter duet values and draw 
    locations = pd.read_csv('draw_large.csv')
    duet_results_3 = full_largest[full_largest.pro1.isin(best_pros_3) & full_largest.pro2.isin(best_pros_3)]
    #make_graphviz_graph_fixed_location(duet_results_3, locations, 'orthogonal_all_coils.svg')
    val_comp, all_vals = draw_num_nans_perm_test(full_largest, n_samples, all_pros, duet_pros)
    boxes = sns.displot(all_vals, height=2, aspect=2)
    plt.axvline(val_comp)
    plt.savefig('orthogonal_duet_all_random_nans.svg')
    plt.close()

    on_targets = "#ffcc00ff"
    off_targets = "#782167ff"
    overall_min = min([min(gap), min(gap_2), min(gap_3)])
    overall_max = max([max(gap), max(gap_2), max(gap_3)])
    small = 3
    big = 5
    linesize = 1
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(gap, size_set, color = off_targets, marker = '.', markersize = small, alpha = 0.75, lw = linesize)
    ax1.plot(special_gap, 9, color = on_targets, marker = 's', markersize = big, alpha = 1)
    ymin_1, ymax_1 = ax1.get_ylim()
    ax1.vlines([bcl_diff, jerala_diff],ymin_1,ymax_1, color = 'gray', ls = 'dotted')
    ax1.set_ylim(ymin_1, ymax_1)
    ax2.plot(gap_2, size_set_2, color = off_targets, marker = '.',  markersize = small, alpha = 0.75,lw = linesize)
    ax2.plot(special_gap_2, 6, color = on_targets, marker = 's', markersize = big, alpha = 1)
    ymin_1, ymax_1 = ax2.get_ylim()
    ax2.vlines([bcl_diff, jerala_diff],ymin_1,ymax_1, color = 'gray', ls = 'dotted')
    ax2.set_ylim(ymin_1, ymax_1)
    #ax2.vlines([bcl_diff, jerala_diff],ymin,ymax, color = 'gray', ls = 'dotted')
    ax3.plot(gap_3, size_set_3, color = off_targets, marker = '.',  markersize = small, alpha = 0.5)
    ax3.plot(special_gap_3, 16, color = on_targets, marker = 's', markersize = big, alpha = 1,lw = linesize)
    #ax3.vlines([bcl_diff, jerala_diff],ymin,ymax, color = 'gray', ls = 'dotted')
    ymin_1, ymax_1 = ax3.get_ylim()
    ax3.vlines([bcl_diff, jerala_diff],ymin_1,ymax_1, color = 'gray', ls = 'dotted')
    ax3.set_ylim(ymin_1, ymax_1)

    plt.xlim(overall_min,overall_max)
    fig.tight_layout()
    fig.set_size_inches(1,2.25)
    plt.savefig('duet_supersets_all.svg', dpi = 300)
    plt.close()

if __name__ == "__main__":
    main()