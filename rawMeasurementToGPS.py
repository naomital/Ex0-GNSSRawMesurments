import csv
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyproj
import simplekml

from ephemeris_manager import EphemerisManager

WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8
transformer = pyproj.Transformer.from_crs(
    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
    {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
)


def get_measurements(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Fix' in row[0]:
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    measurements = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measurements.append(row[1:])

    android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
    measurements = pd.DataFrame(measurements[1:], columns=measurements[0])

    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'

    ## check for types 5 6
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Remove all non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # Convert columns to numeric representation
    measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
    measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
    measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
    measurements['ReceivedSvTimeNanos'] = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
    measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
    measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

    # A few measurement values are not provided by all phones
    # We'll check for them and initialize them with zeros if missing
    if 'BiasNanos' in measurements.columns:
        measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos'])
    else:
        measurements['BiasNanos'] = 0
    if 'TimeOffsetNanos' in measurements.columns:
        measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos'])
    else:
        measurements['TimeOffsetNanos'] = 0

    measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (
            measurements['FullBiasNanos'] - measurements['BiasNanos'])
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)
    measurements['UnixTime'] = measurements['UnixTime']

    # Split data into measurement epochs
    measurements['Epoch'] = 0
    measurements.loc[
        measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum()

    # This should account for rollovers since it uses a week number specific to each measurement
    measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (
            measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
    measurements['tRxSeconds'] = 1e-9 * measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
    measurements['tTxSeconds'] = 1e-9 * (measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
    # Calculate pseudorange in seconds
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']

    # Conver to meters
    measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
    measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']

    measurements = measurements[measurements['SvName'].notna()]
    return measurements


def calculate_satellite_position(ephemeris, transmit_time):
    mu = 3.986005e14
    OmegaDot_e = 7.2921151467e-5
    F = -4.442807633e-10
    sv_position = pd.DataFrame()
    sv_position['sv'] = ephemeris.index
    sv_position.set_index('sv', inplace=True)
    sv_position['t_k'] = transmit_time - ephemeris['t_oe']
    A = ephemeris['sqrtA'].pow(2)
    n_0 = np.sqrt(mu / A.pow(3))
    n = n_0 + ephemeris['deltaN']
    M_k = ephemeris['M_0'] + n * sv_position['t_k']
    E_k = M_k
    err = pd.Series(data=[1] * len(sv_position.index))
    i = 0
    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e'] * np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i += 1

    sinE_k = np.sin(E_k)
    cosE_k = np.cos(E_k)
    delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
    delT_oc = transmit_time - ephemeris['t_oc']
    sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris[
        'SVclockDriftRate'] * delT_oc.pow(2)

    v_k = np.arctan2(np.sqrt(1 - ephemeris['e'].pow(2)) * sinE_k, (cosE_k - ephemeris['e']))

    Phi_k = v_k + ephemeris['omega']

    sin2Phi_k = np.sin(2 * Phi_k)
    cos2Phi_k = np.cos(2 * Phi_k)

    du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k
    dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
    di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k

    u_k = Phi_k + du_k

    r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k

    i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['t_k']

    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)

    Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['t_k'] - OmegaDot_e * ephemeris[
        't_oe']

    sv_position['x_sat'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
    sv_position['y_sat'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
    sv_position['z_sat'] = y_k_prime * np.sin(i_k)

    return sv_position


def least_squares_theres(xs, measured_pseudorange, x0, b0):
    dx = 100 * np.ones(3)
    b = b0
    # set up the G matrix with the right dimensions. We will later replace the first 3 columns
    # note that b here is the clock bias in meters equivalent, so the actual clock bias is b/LIGHTSPEED
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0
    while np.linalg.norm(dx) > 1e-3:
        # Eq. (2):
        r = np.linalg.norm(xs - x0, axis=1)
        # Eq. (1):
        phat = r + b0
        # Eq. (3):
        deltaP = measured_pseudorange - phat
        G[:, 0:3] = -(xs - x0) / r[:, None]
        # Eq. (4):
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
        # Eq. (5):
        dx = sol[0:3]
        db = sol[3]
        x0 = x0 + dx
        b0 = b0 + db
    return x0, b0


def least_squares_direction(xs, measured_pseudorange, x0, b0):
    change = 2
    for _ in range(200):
        changed = False
        error_best = np.linalg.norm(measured_pseudorange - np.linalg.norm(xs - x0, axis=1) + b0)
        dx = error_best / change
        best_dx = np.array([1, 1, 1])
        for dx_try in [(dx, 0, 0), (-dx, 0, 0), (0, dx, 0), (0, -dx, 0), (0, 0, dx), (0, 0, -dx)]:
            error_try = np.linalg.norm(measured_pseudorange - np.linalg.norm(xs - (x0 + dx_try), axis=1) + b0)
            if error_try < error_best:
                error_best = error_try
                best_dx = np.array(dx_try)
                changed = True
        if not changed:
            change += 1
        x0 += best_dx
        b0 = error_best / 10
        if np.linalg.norm(best_dx) <= 1e-3:
            break
    return x0, b0

def find_sat_location(filename, measurements):
    manager = EphemerisManager(filename)
    df_mid_point = pd.DataFrame(
        columns=['sat_name', 'UnixTime', 'GpsTimeNanos', 'tTxSeconds', 'Cn0DbHz', 'PrM', 'delT_sv', 'Epoch', 'x_sat',
                 'y_sat',
                 'z_sat', ])
    for epoch in measurements['Epoch'].unique():
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch)]
        one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')
        timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        sats = one_epoch.index.unique().tolist()
        ephemeris = manager.get_ephemeris(timestamp, sats)
        sv_position = calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

        combo = pd.concat([one_epoch, sv_position], axis=1).reindex()
        combo['sat_name'] = combo['Constellation'] + combo['Svid']
        small = combo[
            ['sat_name', 'UnixTime', 'GpsTimeNanos', 'tTxSeconds', 'Cn0DbHz', 'PrM', 'delT_sv', 'Epoch', 'x_sat',
             'y_sat', 'z_sat']]
        if len(df_mid_point) == 0:
            df_mid_point = small
        else:
            df_mid_point = pd.concat([df_mid_point, small])

    csv_file_name = filename + "_mid_point.csv"
    df_mid_point.to_csv(csv_file_name, index=False)
    return csv_file_name


def find_earth_location_all(csv_file_name):
    measurements = pd.read_csv(csv_file_name)
    df_ans = pd.DataFrame(
        columns=['sat_name', 'UnixTime', 'GpsTimeNanos', 'tTxSeconds', 'Cn0DbHz', 'PrM', 'delT_sv', 'Epoch', 'x_sat',
                 'y_sat', 'z_sat', 'pos_x', 'pos_y', 'pos_z', 'lat', 'lon', 'alt'])
    ecef_list = []
    b0 = 0
    x0 = np.array([0.0, 0.0, 0.0])
    for epoch in measurements['Epoch'].unique():
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch)]
        one_epoch = one_epoch.drop_duplicates(subset='sat_name')
        if len(one_epoch.index) > 4:
            xs = one_epoch[['x_sat', 'y_sat', 'z_sat']].to_numpy()
            pr = one_epoch['PrM'] + LIGHTSPEED * one_epoch['delT_sv']
            pr = pr.to_numpy()
            x, b = least_squares_direction(xs, pr, x0, b0)
            b0 = b
            x0 = x
            ecef_list.append(x)
            lon, lat, alt = transformer.transform(x[0], x[1], x[2], radians=False)
            one_epoch['pos_x'] = x[0]
            one_epoch['pos_y'] = x[1]
            one_epoch['pos_z'] = x[2]
            one_epoch['lat'] = lat
            one_epoch['lon'] = lon
            one_epoch['alt'] = alt

            if len(df_ans) == 0:
                df_ans = one_epoch
            else:
                df_ans = pd.concat([df_ans, one_epoch])

    short = df_ans.drop_duplicates(['UnixTime'])
    kml = simplekml.Kml()
    for row, col in short.iterrows():
        kml.newpoint(name=str(row), coords=[(col['lon'], col['lat'])])
    kml_file_name = csv_file_name.split('.')[0]
    if '_mid_point' in kml_file_name:
        kml_file_name = kml_file_name.replace('_mid_point', '')
    kml_file_name += '.kml'
    kml.save(kml_file_name)
    new_csv_file_name = csv_file_name.split('.')[0]
    if '_mid_point' in new_csv_file_name:
        new_csv_file_name = new_csv_file_name.replace('_mid_point', '')
    new_csv_file_name += '_answer.csv'

    df_ans.to_csv(new_csv_file_name, index=False)


def main():
    filename = ''
    data = get_measurements(filename)
    csv_file_name = find_sat_location(filename.split('.')[0], data)
    find_earth_location_all(csv_file_name)


if __name__ == "__main__":
    main()
