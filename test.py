import copy
def build_single_interaction_factor(metadata, design_factor, continuous_factors):
    """
    Create an interaction factor from two design factors. This function supports
    categorical and continuous factors.

    Parameters
    ----------
    metadata: pandas.DataFrame
        The dataframe to modify.
    design_factor: str
        The name of the design factor to split.
    continuous_factors: list
        The list of continuous factors.
    """
    if isinstance(design_factor, str):
        if ":" in design_factor:
            final_name = copy.deepcopy(design_factor)
            design_factor = design_factor.split(':')
            if not all([des_f in metadata.columns for des_f in design_factor]):
                raise ValueError(f"Some of the design factors in {design_factor} are not in metadata")
            
        else:
            # We leave the dataframe metadata unchanged
            return

    nfactors = len(design_factor)
    if nfactors > 2:
        m = nfactors // 2
        # function modifies metadata, continuous_factors inplace
        left_part = build_single_interaction_factor(metadata, design_factor[:m], continuous_factors)
        right_part = build_single_interaction_factor(metadata, design_factor[m:], continuous_factors)
        return build_single_interaction_factor(metadata, left_part + right_part, continuous_factors)

    if nfactors == 2:
        left_factor, right_factor = design_factor[0], design_factor[1] 
        interaction_column_name = left_factor + ":" + right_factor
        is_left_categorical = left_factor not in continuous_factors
        is_right_categorical = right_factor not in continuous_factors
        is_any_categorical = is_left_categorical or is_right_categorical
        if not is_any_categorical:
            # All factors are continuous
            metadata[interaction_column_name] = metadata[left_factor] * metadata[right_factor]
            continuous_factors.append(interaction_column_name)
            return [interaction_column_name]
        else:
            if is_left_categorical and is_right_categorical:
                merge_categorical_columns_inplace(metadata, left_factor, right_factor)
                return [interaction_column_name]

            elif is_left_categorical:
                cat_col_name = left_factor
                cont_col_name = right_factor
            elif is_right_categorical:
                cat_col_name = right_factor
                cont_col_name = left_factor
            
            cat_levels = metadata[cat_col_name].unique()
            # We multiplex the continuous variable to cat_levels continuous variables
            # that are cont_col_name when the level is activated and 0 otherwise
            import pdb; pdb.set_trace()
            interaction_column_names = []
            for idx, cat_level in enumerate(cat_levels):
                current_col_name = interaction_column_name + f"_{idx}"
                interaction_column_names.append(current_col_name)
                metadata[current_col_name] = (metadata[cat_col_name] == cat_level).astype("float") * metadata[cont_col_name]
                continuous_factors.append(current_col_name)
            import pdb; pdb.set_trace()
            return interaction_column_names
        
    elif nfactors == 1:
        return [design_factor[0]]

import pandas as pd 
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
build_single_interaction_factor(df, "a:b:c", continuous_factors=["c"])
print(df)