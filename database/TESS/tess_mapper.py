from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
import pandas as pd

def get_tic_id_from_gaia(gaia_id):
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('ids')

    result = custom_simbad.query_object(f"Gaia DR3 {gaia_id}")
    if result is None or result['IDS'].mask.any():
        return None

    ids = result['IDS'][0]
    for id_str in ids.split('|'):
        if 'TIC' in id_str:
            return id_str.split(' ')[1]

    return None

def get_tic_parameters(tic_id):
    catalog_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
    if len(catalog_data) == 0:
        return None

    parameters = {
        'tic_id': tic_id,
        'ra': catalog_data['ra'][0] if 'ra' in catalog_data.colnames else None,
        'dec': catalog_data['dec'][0] if 'dec' in catalog_data.colnames else None,
        'plx': catalog_data['plx'][0] if 'plx' in catalog_data.colnames else None,
        'pmRA': catalog_data['pmRA'][0] if 'pmRA' in catalog_data.colnames else None,
        'pmDEC': catalog_data['pmDEC'][0] if 'pmDEC' in catalog_data.colnames else None,
        'Teff': catalog_data['Teff'][0] if 'Teff' in catalog_data.colnames else None,
        'logg': catalog_data['logg'][0] if 'logg' in catalog_data.colnames else None,
        'MH': catalog_data['MH'][0] if 'MH' in catalog_data.colnames else None,
        'rad': catalog_data['rad'][0] if 'rad' in catalog_data.colnames else None,
        'lum': catalog_data['lum'][0] if 'lum' in catalog_data.colnames else None,
        'gallong': catalog_data['gallong'][0] if 'gallong' in catalog_data.colnames else None,
        'gallat': catalog_data['gallat'][0] if 'gallat' in catalog_data.colnames else None,
        'Tmag': catalog_data['Tmag'][0] if 'Tmag' in catalog_data.colnames else None,
        'Vmag': catalog_data['Vmag'][0] if 'Vmag' in catalog_data.colnames else None,
        'GAIAmag': catalog_data['GAIAmag'][0] if 'GAIAmag' in catalog_data.colnames else None
    }

    return parameters

def map_gaia_to_tic_and_retrieve_parameters(gaia_ids):
    tic_data_list = []
    total_ids = len(gaia_ids)

    for idx, gaia_id in enumerate(gaia_ids):
        print(f"Processing {idx + 1}/{total_ids} objects...")
        tic_id = get_tic_id_from_gaia(gaia_id)
        if tic_id is None:
            tic_data_list.append({'source_id': gaia_id, 'tic_id': None})
            continue

        tic_parameters = get_tic_parameters(tic_id)
        if tic_parameters:
            tic_parameters['source_id'] = gaia_id
            tic_data_list.append(tic_parameters)
        else:
            tic_data_list.append({'source_id': gaia_id, 'tic_id': tic_id})

    return tic_data_list

coralie_df = pd.read_csv('../coralie-result.csv')
gaia_ids = coralie_df['source_id'].tolist()
tic_data_list = map_gaia_to_tic_and_retrieve_parameters(gaia_ids)
tic_df = pd.DataFrame(tic_data_list)
tic_df.to_csv('coralie_tic.csv', index=False)

print(tic_df.head())
