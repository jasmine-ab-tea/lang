import pandas as pd
import numpy as np

def generate_experiment_data(num_users: int) -> pd.DataFrame:
    # Generate unique entity IDs
    entity_ids = np.arange(1, num_users + 1)
    
    # Randomly assign users to experiment groups (0 for control, 1 for treatment)
    experiment_group = np.random.choice([0, 1], size=num_users)
    
    # Generate conversion rates
    # Control group has a base conversion rate
    control_conversion_rate = np.random.uniform(0.1, 0.3, size=num_users)
    # Treatment group has a slightly higher conversion rate
    treatment_conversion_rate = control_conversion_rate + np.random.uniform(0.05, 0.1, size=num_users)
    
    # Apply conversion rates based on experiment group
    conversion_rate = np.where(experiment_group == 0, control_conversion_rate, treatment_conversion_rate)
    
    # Randomly assign countries
    countries = np.random.choice(['US', 'CA', 'UK'], size=num_users)
    
    # Randomly assign delivery types
    delivery_types = np.random.choice(['ASAP', 'scheduled'], size=num_users)
    
    # Create the DataFrame
    data = pd.DataFrame({
        'entity_id': entity_ids,
        'conversion_rate': conversion_rate,
        'experiment_group': experiment_group,
        'country': countries,
        'delivery_type': delivery_types
    })
    
    return data

df = generate_experiment_data(100)
df.to_csv('./smart_ds/experiment_data.csv', index=False)
