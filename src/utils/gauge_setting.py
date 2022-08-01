from cgitb import handler
import numpy as np


def find_top_5_nearest(array, value):
    """Find the top 5 closest neighbors of given optimal value

    Args:
        array (np.array): the generated data with prediction
        value (int or float): optimal value to find

    Returns:
        list: the top 5 indices of suggested value
    """
    array = np.asarray(array)
    diff = np.abs(array - value)
    indices = np.argsort(diff)
    return indices


schrott_preis = {"Legierung_000004_FE_MO":16.41,"Legierung_000001FE_CRCARBURE":0.68,"Legierung_000016FE_CR006":2.4,	"Legierung_000003_FE_SI":1.36,"Legierung_000013_FE_MANGANCARBURE":1.06,	"Legierung_000005_FE_V80":21.85,"Legierung_000018_KOHLE":0.86,	"Legierung_000007_FE_W75_80":19.89,	"Fremdschrott_000044_CHROMPAKETE":0.62,"Fremdschrott_000001_HACKSCHROTT":0.38,	"Fremdschrott_000072_HACKSCHROTT":0.4,	"Fremdschrott_000034_2343":0.38,"Fremdschrott_000013_2367":0.83,"Fremdschrott_000029_2344":0.65,"Fremdschrott_000048_2343":0.62,"Fremdschrott_000072_Manganarm":0.4,"Fremdschrott_000021_2365":0.92,"Fremdschrott_000516_2379":0.73,	"Fremdschrott_000063_2345":1.6,	"Fremdschrott_000513_2367":	0.74,"Fremdschrott_000534_2343":0.57,"Fremdschrott_000043_Hackschrott":	0.38,"Fremdschrott_000109_2379":0.8,"Fremdschrott_000006_CHROM_SCHROTT":3.2,"Fremdschrott_000049_2344":0.62,"Fremdschrott_000112_CR_SCHROTT":0.8,"Fremdschrott_000114_Chrom_Schrott":0.8,"Fremdschrott_000041_2379":0.53,"Fremdschrott_000563_2345":0.65,"Fremdschrott_000516_2379.1":0.67, "Fremdschrott_000564_2895":0.98,"Fremdschrott_000044_ChromSchrott":0.62,"Fremdschrott_000004_3343":3.2,	"Fremdschrott_000002_3355":0.38,"Fremdschrott_000073_3343":0.4,	"Fremdschrott_000043_KUPOLSCHROTT":0.38,"Kreislauf_000437_4112":3.35,"Kreislauf_000031_2343":0.32,"Kreislauf_000039_2367":0.51,	"Kreislauf_000032_2344":0.3, "Kreislauf_000043_2379":0.42,"Kreislauf_000052_UMSCHMELZBLOCK":0.39,	"Kreislauf_000072_2601":0.17,"Kreislauf_000163_2897":0.68,	"Kreislauf_000071_4112":0.38,"Kreislauf_000175_4192":0.44,	"Kreislauf_000326_4197":0.68,"Kreislauf_000269_2360":0.45,"Kreislauf_000121_2895":0.84,	"Kreislauf_000087_2695":0.91,	"Kreislauf_000161_4034":0.68, "Kreislauf_000164_2345":0.69,	"Kreislauf_000035_2360":4.6,"Kreislauf_000171_4125":0.4,	"Kreislauf_000063_2519": 0.41,	"Kreislauf_000009_2080": 0.25,	"Kreislauf_000300_2897": 0.2,"Kreislauf_000034_2357":0.35,"Kreislauf_000033_2345": 0.32, "Kreislauf_000038_2365": 0.46,	"Kreislauf_000393_4057":0.82,	"Kreislauf_000281_2360": 0.58,	"Kreislauf_000389_4021": 0.5,"Kreislauf_000045_2390": 0.46,	"Kreislauf_000345_2419":0.47,"Kreislauf_000415_2389":0.6,	"Kreislauf_000051_4125": 0.35,"Kreislauf_000144_3343":1.83,	"Kreislauf_000148_3355":1.28,"Kreislauf_000145_3344":1.55,	"Kreislauf_000141_3333":1.17,"Kreislauf_000179_3344":0.57
}


def calculate_schrott(df):
    """Calculate the Schrott value of each product"""
    columns = df.columns.tolist()

    energie = df.iloc[0]['Prediction']

    fremdschrott_columns = [column for column in columns if 'Fremdschrott' in column]
    legierung_columns = [column for column in columns if 'Legierung' in column]
    kreislauf_columns = [column for column in columns if 'Kreislauf' in column]
    
    fremdschrott_menge = df.iloc[0][fremdschrott_columns].sum()
    legierung_menge = df.iloc[0][legierung_columns].sum()
    kreislauf_menge = df.iloc[0][kreislauf_columns].sum()

    
    return energie, fremdschrott_menge, legierung_menge, kreislauf_menge

def calculate_original_schrott(df, energie):
    columns = df.columns.tolist()
    energie_column = df['Energy']
    indices = find_top_5_nearest(energie_column, energie)
    optimal_value = energie_column[indices[0]]
    
    fremdschrott_columns = [column for column in columns if 'Fremdschrott' in column]
    legierung_columns = [column for column in columns if 'Legierung' in column]
    kreislauf_columns = [column for column in columns if 'Kreislauf' in column]
    
    fremdschrott_menge = df.iloc[indices[:50]][fremdschrott_columns].sum(axis=1).median()
    legierung_menge = df.iloc[indices[:50]][legierung_columns].sum(axis=1).median()
    kreislauf_menge = df.iloc[indices[:50]][kreislauf_columns].sum(axis=1).median()
    
    
    
    return optimal_value, fremdschrott_menge, legierung_menge, kreislauf_menge


def linear_optimization(df_op, df_handler):
    """linear optimization find the best combination of handler who provides lowest price of schrott

    Args:
        df_op (pd.DataFrame): dataframe after gauge optimization contains the amount of every schrott 
        df_handler (pd.DataFrame): dataframe with the handler data and their price 
    """
    from ortools.linear_solver import pywraplp

    # create the constraint matrix for gattierungen
    legierung = [column for column in df_op.columns.tolist() if 'Legierung' in column]
    fremd = [column for column in df_op.columns.tolist() if 'Fremdschrott' in column]
    zu_kauf = legierung + fremd
    gattierung = df_op.iloc[0][zu_kauf]
    gattierung = gattierung.to_frame().reset_index()
    gattierung = gattierung.values.tolist()
    
    # data contains the hander information and their price
    data = df_handler.values.tolist()
    
    # create the google or tools sovler
    solver = pywraplp.Solver('Gattierungplan', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    schrotte = [solver.NumVar(0.0, solver.infinity(), item[0]) for item in data]
    
    constraints = []
    for i, gatt in enumerate(gattierung):
        constraints.append(solver.Constraint(gatt[1], solver.infinity()))
        for j, item in enumerate(data):
            constraints[i].SetCoefficient(schrotte[j], item[i+2])
    
    # objective function: Minimize the price of sum of schrott
    objective = solver.Objective()
    for schrott in schrotte:
        objective.SetCoefficient(schrott, 1)
        
    objective.SetMinimization()
    status = solver.Solve()
    
    msg = ''
    # Check that the problem has an optimal solution.
    if status != solver.OPTIMAL:
        msg = 'The problem does not have an optimal solution!'
        if status == solver.FEASIBLE:
            msg = 'A potentially suboptimal solution was found.'
        else:
            msg = 'The solver could not solve the problem!'
            exit(1)
            
    gattierung_result = [0] * len(gattierung)
    handler_result = []
    for i, schrott in enumerate(schrotte):
        if schrott.solution_value() > 0:
            handler_result.append({data[i][0]: schrott.solution_value()})
            for j, _ in enumerate(gattierung):
                gattierung_result[j] += data[i][j+2] * schrott.solution_value()
    
    total_value = objective.Value()
    # the concreted solution for every gattinerung
    gatt_result = []
    # construct result for displaying: {gattierung: [actual_find_value, min_value]}
    for i, gatt in enumerate(gattierung):
        # only show the gattierungs which are used
        if gatt[1] >0:
            gatt_result.append({gatt[0]: gattierung_result[i]})
        
    return msg, handler_result, total_value, gatt_result
        
        
    
    
    