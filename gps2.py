import sys, os, csv

from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import navpy
import simplekml
import polycircles
from polycircles import polycircles

from ephemeris_manager import EphemerisManager
import pyproj

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


def calc_dist(sat, guess):
    return np.sqrt((sat[0] - guess[0]) ** 2 + (sat[1] - guess[1]) ** 2 + (sat[2] - guess[2]) ** 2)





def least_squares_theres(xs, measured_pseudorange, x0, b0):
    dx = 100*np.ones(3)
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
    norm_dp = np.linalg.norm(deltaP)
    return x0, b0, norm_dp





def earth_location1(sats, pr, x):
    if (x == np.array([0, 0, 0])).all():
        epoch = 50
        lr = 0.5
    else:
        epoch = 10
        lr = 2
    avg_dist = []
    count = 0
    m_best = np.inf
    avg_dist.append(m_best)
    while count < epoch:
        diffs = np.zeros(len(sats))
        for i in range(len(sats)):
            d = calc_dist(sats[i], x)
            diffs[i] = np.abs(pr[i] - d)
        m_best = np.abs(diffs.mean())
        if m_best < avg_dist[-1]:
            avg_dist.append(m_best)

        # we have baseline for this iteration
        # find which direction is best +- xyz
        # change that one

        changes = []
        x2 = x.copy()
        x2[0] = x[0] + avg_dist[-1] / (2 * lr)
        for i in range(len(sats)):
            d = calc_dist(sats[i], x2)
            diffs[i] = np.abs(pr[i] - d)
        m = diffs.mean()
        changes.append(('x', '+', m))

        x2 = x.copy()
        x2[0] = x[0] - avg_dist[-1] / (2 * lr)
        for i in range(len(sats)):
            d = calc_dist(sats[i], x2)
            diffs[i] = np.abs(pr[i] - d)
        m = diffs.mean()
        changes.append(('x', '-', m))

        x2 = x.copy()
        x2[1] = x[1] + avg_dist[-1] / (2 * lr)
        for i in range(len(sats)):
            d = calc_dist(sats[i], x2)
            diffs[i] = np.abs(pr[i] - d)
        m = diffs.mean()
        changes.append(('y', '+', m))

        x2 = x.copy()
        x2[1] = x[1] - avg_dist[-1] / (2 * lr)
        for i in range(len(sats)):
            d = calc_dist(sats[i], x2)
            diffs[i] = np.abs(pr[i] - d)
        m = diffs.mean()
        changes.append(('y', '-', m))

        x2 = x.copy()
        x2[2] = x[2] + avg_dist[-1] / (2 * lr)
        for i in range(len(sats)):
            d = calc_dist(sats[i], x2)
            diffs[i] = np.abs(pr[i] - d)
        m = diffs.mean()
        changes.append(('z', '+', m))

        x2 = x.copy()
        x2[2] = x[2] - avg_dist[-1] / (2 * lr)
        for i in range(len(sats)):
            d = calc_dist(sats[i], x2)
            diffs[i] = np.abs(pr[i] - d)
        m = diffs.mean()
        changes.append(('z', '-', m))
        x2 = 0
        changes = sorted(changes, key=lambda a: a[2])

        if changes[0][2] < avg_dist[-1]:
            if changes[0][1] == '+':
                if changes[0][0] == 'x':
                    x[0] = x[0] + avg_dist[-1] / (2 * lr)
                elif changes[0][0] == 'y':
                    x[1] = x[1] + avg_dist[-1] / (2 * lr)
                elif changes[0][0] == 'z':
                    x[2] = x[2] + avg_dist[-1] / (2 * lr)
            else:
                if changes[0][0] == 'x':
                    x[0] = x[0] - avg_dist[-1] / (2 * lr)
                elif changes[0][0] == 'y':
                    x[1] = x[1] - avg_dist[-1] / (2 * lr)
                elif changes[0][0] == 'z':
                    x[2] = x[2] - avg_dist[-1] / (2 * lr)
            avg_dist.append(changes[0][2])
        else:
            lr = lr + lr / 10
        count += 1

    b0 = 0
    norm_dp = 0

    return x, b0, norm_dp, avg_dist[-1]


def change1(sats, x, avg_dist, lr, pr):

    diffs = np.zeros(len(sats))
    changes = []

    x2 = x.copy()
    x2[0] = x[0] + avg_dist[-1] / (2 * lr)
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('x', '+', m))

    x2 = x.copy()
    x2[0] = x[0] - avg_dist[-1] / (2 * lr)
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('x', '-', m))

    x2 = x.copy()
    x2[1] = x[1] + avg_dist[-1] / (2 * lr)
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('y', '+', m))

    x2 = x.copy()
    x2[1] = x[1] - avg_dist[-1] / (2 * lr)
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('y', '-', m))

    x2 = x.copy()
    x2[2] = x[2] + avg_dist[-1] / (2 * lr)
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('z', '+', m))

    x2 = x.copy()
    x2[2] = x[2] - avg_dist[-1] / (2 * lr)
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('z', '-', m))

    x2 = 0
    changes = sorted(changes, key=lambda a: a[2])

    if changes[0][2] < avg_dist[-1]:
        if changes[0][1] == '+':
            if changes[0][0] == 'x':
                x[0] = x[0] + avg_dist[-1] / (2 * lr)
            elif changes[0][0] == 'y':
                x[1] = x[1] + avg_dist[-1] / (2 * lr)
            elif changes[0][0] == 'z':
                x[2] = x[2] + avg_dist[-1] / (2 * lr)
        else:
            if changes[0][0] == 'x':
                x[0] = x[0] - avg_dist[-1] / (2 * lr)
            elif changes[0][0] == 'y':
                x[1] = x[1] - avg_dist[-1] / (2 * lr)
            elif changes[0][0] == 'z':
                x[2] = x[2] - avg_dist[-1] / (2 * lr)
        avg_dist.append(changes[0][2])
    else:
        lr = lr + lr / 10

    return avg_dist, x, lr

def change2 (sats, x, avg_dist, lr, pr):
    diffs = np.zeros(len(sats))
    changes = []

    x2 = x.copy()
    x2[0] = x[0] + lr
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('x', '+', m))

    x2 = x.copy()
    x2[0] = x[0] - lr
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('x', '-', m))

    x2 = x.copy()
    x2[1] = x[1] + lr
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('y', '+', m))

    x2 = x.copy()
    x2[1] = x[1] - lr
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('y', '-', m))

    x2 = x.copy()
    x2[2] = x[2] + lr
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('z', '+', m))

    x2 = x.copy()
    x2[2] = x[2] - lr
    for i in range(len(sats)):
        d = calc_dist(sats[i], x2)
        diffs[i] = np.abs(pr[i] - d)
    m = diffs.mean()
    changes.append(('z', '-', m))
    x2 = 0
    changes = sorted(changes, key=lambda a: a[2])

    if changes[0][2] < avg_dist[-1]:
        if changes[0][1] == '+':
            if changes[0][0] == 'x':
                x[0] = x[0] + lr
            elif changes[0][0] == 'y':
                x[1] = x[1] + lr
            elif changes[0][0] == 'z':
                x[2] = x[2] + lr
        else:
            if changes[0][0] == 'x':
                x[0] = x[0] - lr
            elif changes[0][0] == 'y':
                x[1] = x[1] - lr
            elif changes[0][0] == 'z':
                x[2] = x[2] - lr
        avg_dist.append(changes[0][2])
    else:
        lr -= 0.1

    return avg_dist, x, lr


def earth_location2(sats, pr, x):
    if (x == np.array([0, 0, 0])).all():
        epoch = 50
        lr = 0.5
    else:
        epoch = 10
        lr = 2
    lr2 = 1
    avg_dist = []
    count = 0
    m_best = np.inf
    avg_dist.append(m_best)
    while count < epoch:
        if len(avg_dist)> 2 and avg_dist[-2]-avg_dist[-1] < 0.05:
            break
        diffs = np.zeros(len(sats))
        for i in range(len(sats)):
            d = calc_dist(sats[i], x)
            diffs[i] = np.abs(pr[i] - d)
        m_best = np.abs(diffs.mean())
        if m_best < avg_dist[-1]:
            avg_dist.append(m_best)

        if lr < 4:
            avg_dist, x, lr = change1(sats,x,avg_dist,lr,pr)
        else:
            avg_dist, x, lr2 = change2(sats, x, avg_dist, lr2, pr)

        count += 1
    print("it took ", count -1, " iterations")
    b0 = 0
    norm_dp = 0

    return x, b0, norm_dp, avg_dist[-1]


#
# def least_squares1(xs, pr, x, b):
#     # this function tries to find the location using only xyz
#
#     # i think b is the time dalay that is also not known
#     if (x == np.array([0, 0, 0])).all():
#         epoch = 100
#     else:
#         epoch = 40
#     avg_dist = []
#     count = 0
#     m_best = np.inf
#     avg_dist.append(m_best)
#     while count < epoch:
#         diffs = np.zeros(len(xs))
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x)
#             diffs[i] = np.abs(pr[i] - d)
#         m_best = np.abs(diffs.mean())
#         if m_best < avg_dist[-1]:
#             avg_dist.append(m_best)
#
#
#         # we have baseline for this iteration
#         # find which direction is best +- xyz
#         # change that one
#
#         changes = []
#         x2 = x.copy()
#         x2[0] = x[0] + 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('x', '+', m))
#         x2 = x.copy()
#         x2[0] = x[0] - 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('x', '-', m))
#         x2 = x.copy()
#         x2[1] = x[1] + 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('y', '+', m))
#         x2 = x.copy()
#         x2[1] = x[1] - 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('y', '-', m))
#         x2 = x.copy()
#         x2[2] = x[2] + 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('z', '+', m))
#         x2 = x.copy()
#         x2[2] = x[2] - 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('z', '-', m))
#
#         changes = sorted(changes, key=lambda a: a[2])
#         x2 = x.copy()
#         if changes[0][1] == '+':
#             if changes[0][0] == 'x':
#                 x2[0] = x[0] + avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             elif changes[0][0] == 'y':
#                 x2[1] = x[1] + avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             else:
#                 x2[2] = x[2] + avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#         else:
#             if changes[0][0] == 'x':
#                 x2[0] = x[0] - avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             elif changes[0][0] == 'y':
#                 x2[1] = x[1] - avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             else:
#                 x2[2] = x[2] - avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#         if m < avg_dist[-1]:
#             avg_dist.append(m)
#             x = x2
#
#         count += 1
#
#     b0 = 0
#     norm_dp = 0
#
#     return x, b0, norm_dp, avg_dist[-1]
#
#
# def least_squares2(xs, pr, x, b, t):
#     # this function tries to find the location using xyz and t but not sure if using t is wrong
#     # i think b is the time dalay that is also not known
#
#     for i in range(len(t)):
#         t[i] = t[i] / LIGHTSPEED
#     if (x == np.array([0, 0, 0])).all():
#         epoch = 200
#     else:
#         epoch = 40
#     avg_dist = []
#     count = 0
#     change_by = 2
#     didnt_change = 0
#     m_best = np.inf
#     avg_dist.append(m_best)
#     while count < epoch:
#         diffs = np.zeros(len(xs))
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x)
#             d = d - t[i]
#             diffs[i] = np.abs(pr[i] - d)
#         m_best = np.abs(diffs.mean())
#         if m_best < avg_dist[-1]:
#             avg_dist.append(m_best)
#         # we have baseline for this iteration
#         # find which direction is best +- xyz
#         # change that one
#
#         changes = []
#         x2 = x.copy()
#         x2[0] = x[0] + 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             d = d - t[i]
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('x', '+', m))
#         x2 = x.copy()
#         x2[0] = x[0] - 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             d = d - t[i]
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('x', '-', m))
#         x2 = x.copy()
#         x2[1] = x[1] + 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             d = d - t[i]
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('y', '+', m))
#         x2 = x.copy()
#         x2[1] = x[1] - 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             d = d - t[i]
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('y', '-', m))
#         x2 = x.copy()
#         x2[2] = x[2] + 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             d = d - t[i]
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('z', '+', m))
#         x2 = x.copy()
#         x2[2] = x[2] - 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             d = d - t[i]
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('z', '-', m))
#
#         changes = sorted(changes, key=lambda a: a[2])
#
#         x2 = x.copy()
#         if changes[0][1] == '+':
#             if changes[0][0] == 'x':
#                 x2[0] = x[0] + avg_dist[-1] / change_by
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     d = d - t[i]
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             elif changes[0][0] == 'y':
#                 x2[1] = x[1] + avg_dist[-1] / change_by
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     d = d - t[i]
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             else:
#                 x2[2] = x[2] + avg_dist[-1] / change_by
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     d = d - t[i]
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#         else:
#             if changes[0][0] == 'x':
#                 x2[0] = x[0] - avg_dist[-1] / change_by
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     d = d - t[i]
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             elif changes[0][0] == 'y':
#                 x2[1] = x[1] - avg_dist[-1] / change_by
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     d = d - t[i]
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             else:
#                 x2[2] = x[2] - avg_dist[-1] / change_by
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     d = d - t[i]
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#         if m < avg_dist[-1]:
#             avg_dist.append(m)
#             didnt_change = 0
#             x = x2
#         else:
#             didnt_change += 1
#         if didnt_change == 5:
#             change_by += 1
#         count += 1
#
#     # print(avg_dist[-1])
#     # print(x)
#     # print(avg_dist)
#     b0 = 0
#     norm_dp = 0
#     return x, b0, norm_dp, avg_dist[-1]
#
#
# def get_rid_of_sat1(sats, pr, x, total_average):
#     diffs = np.zeros(len(sats))
#     for i in range(len(sats)):
#         d = calc_dist(sats[i], x)
#         diffs[i] = np.abs(pr[i] - d)
#     farthest = []
#     for i in range(len(diffs)):
#         d = np.abs(diffs[i] - total_average)
#         farthest.append((d, i))
#     farthest = sorted(farthest, key=lambda a: a[0], reverse=True)
#     sats = np.delete(sats, farthest[0][1], 0)
#     pr = np.delete(pr, farthest[0][1])
#     return sats, pr
#
# def get_rid_of_sat2(sats, pr, x, total_average):
#     bigger=[]
#     smaller=[]
#     rows = []
#     for i in range(len(sats)):
#         d = calc_dist(sats[i], x)
#         if d > total_average:
#             bigger.append((d, i))
#         else:
#             smaller.append((d, i))
#     if len(bigger)>=4 and len(smaller)>=4:
#         print("making sat group smaller both big")
#         if len(bigger)>len(smaller):
#             for i in smaller:
#                 rows.append(i[1])
#         else:
#             for i in bigger:
#                 rows.append(i[1])
#     elif len(bigger)>=4:
#         print("making sat group smaller, big is big")
#         for i in smaller:
#             rows.append(i[1])
#     elif len(smaller) >= 4:
#         print("making sat group smaller small is big")
#         for i in bigger:
#             rows.append(i[1])
#     sats = np.delete(sats, rows, 0)
#     pr = np.delete(pr, rows)
#     return sats, pr
#
# def get_rid_of_sat3(sats, pr, x, total_average):
#     bigger = []
#     smaller = []
#     rows = []
#     for i in range(len(sats)):
#         d = calc_dist(sats[i], x)
#         if d > total_average:
#             bigger.append((d, i))
#         else:
#             smaller.append((d, i))
#     if len(smaller) >= 4:
#         print("making sat group smaller to small than avg group")
#         for i in bigger:
#             rows.append(i[1])
#     sats = np.delete(sats, rows, 0)
#     pr = np.delete(pr, rows)
#     return sats, pr
#
# def least_squares3(xs, pr, x, b):
#     # this function tries to find the location using only xyz
#     # and cuts out satalites as we get closer
#
#     # i think b is the time dalay that is also not known
#     if (x == np.array([0, 0, 0])).all():
#         epoch = 100
#         getrid = 25
#     else:
#         epoch = 40
#         getrid = 10
#     avg_dist = []
#     count = 0
#     orig = len(xs)
#     threshold= 100000
#     m_best = np.inf
#     avg_dist.append(m_best)
#
#     while count < epoch:
#         diffs = np.zeros(len(xs))
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x)
#             diffs[i] = np.abs(pr[i] - d)
#         m_best = np.abs(diffs.mean())
#         if m_best < avg_dist[-1]:
#             avg_dist.append(m_best)
#
#         # we have baseline for this iteration
#         # find which direction is best +- xyz
#         # change that one
#         changes = []
#         x2 = x.copy()
#         x2[0] = x[0] + 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('x', '+', m))
#         x2 = x.copy()
#         x2[0] = x[0] - 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('x', '-', m))
#         x2 = x.copy()
#         x2[1] = x[1] + 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('y', '+', m))
#         x2 = x.copy()
#         x2[1] = x[1] - 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('y', '-', m))
#         x2 = x.copy()
#         x2[2] = x[2] + 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('z', '+', m))
#         x2 = x.copy()
#         x2[2] = x[2] - 1
#         for i in range(len(xs)):
#             d = calc_dist(xs[i], x2)
#             diffs[i] = np.abs(pr[i] - d)
#         m = diffs.mean()
#         changes.append(('z', '-', m))
#
#         changes = sorted(changes, key=lambda a: a[2])
#         x2 = x.copy()
#         if changes[0][1] == '+':
#             if changes[0][0] == 'x':
#                 x2[0] = x[0] + avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             elif changes[0][0] == 'y':
#                 x2[1] = x[1] + avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             else:
#                 x2[2] = x[2] + avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#         else:
#             if changes[0][0] == 'x':
#                 x2[0] = x[0] - avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             elif changes[0][0] == 'y':
#                 x2[1] = x[1] - avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#             else:
#                 x2[2] = x[2] - avg_dist[-1] / 2
#                 for i in range(len(xs)):
#                     d = calc_dist(xs[i], x2)
#                     diffs[i] = np.abs(pr[i] - d)
#                 m = diffs.mean()
#         if m < avg_dist[-1]:
#             avg_dist.append(m)
#             x = x2
#
#         count += 1
#         if count % getrid == 0 and avg_dist[-1]<threshold:
#             threshold/=10
#             if len(xs) > 4:
#                 # xs, pr = get_rid_of_sat1(xs, pr, x, avg_dist[-1])
#                 xs, pr = get_rid_of_sat3(xs, pr, x, avg_dist[-1])
#
#     b0 = 0
#     norm_dp = 0
#     return x, b0, norm_dp, avg_dist[-1]

def find_sat_location(file, measurements):
    mid_point = []
    manager = EphemerisManager(file.split('.')[0])
    df_mid_point = pd.DataFrame(
        columns=['sat_name', 'UnixTime','GpsTimeNanos', 'tTxSeconds', 'Cn0DbHz', 'PrM', 'delt_sv', 'Epoch', 'x_sat', 'y_sat',
                 'z_sat',])
    for epoch in measurements['Epoch'].unique():
        # one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)]
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch)]
        one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')
        # if len(one_epoch.index) > 4:
        timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        sats = one_epoch.index.unique().tolist()
        ephemeris = manager.get_ephemeris(timestamp, sats)
        sv_position = calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

        combo = pd.concat([one_epoch, sv_position], axis=1).reindex()
        combo['sat_name'] = combo['Constellation'] + combo['Svid']
        # print(combo.columns)
        small = combo[
            ['sat_name', 'UnixTime','GpsTimeNanos', 'tTxSeconds', 'Cn0DbHz', 'PrM', 'delT_sv', 'Epoch', 'x_sat', 'y_sat', 'z_sat']]
        df_mid_point = pd.concat([df_mid_point, small])

    df_mid_point.rename(columns={"sat_name": "sat_name", "UnixTime": "UnixTime",'GpsTimeNanos':'GpsTimeNanos',
                                 'tTxSeconds':'tTxSeconds','Cn0DbHz':'Cn0DbHz','PrM':'PrM',
                                 'delT_sv':'delT_sv',"Epoch":"Epoch",  'x_sat': 'x_sat',
                                 'y_sat':'y_sat', 'z_sat': 'z_sat'
                                 })
    name = file.split('.')[0] + "_mid_point.csv"
    df_mid_point.to_csv(name, index=False)
    return name


def find_earth_location_all(file):
    measurements = pd.read_csv(file)
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    measurements['time_stamp'] = pd.to_datetime(measurements['GpsTimeNanos'],utc=True, origin=gpsepoch)

    # measurements = measurements.loc[50:]

    ans = []
    manager = EphemerisManager(file.split('.')[0])
    ecef_list = []
    final_average_dist_list = []
    b0 = 0
    x0 = np.array([0, 0, 0])
    for epoch in measurements['Epoch'].unique():
        # one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)]
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch)]
        one_epoch = one_epoch.drop_duplicates(subset='sat_name').set_index('sat_name')
        if len(one_epoch.index) > 4:
            timestamp = one_epoch.iloc[0]['time_stamp'].to_pydatetime(warn=False)
            sats = one_epoch.index.unique().tolist()
            ephemeris = manager.get_ephemeris(timestamp, sats)
            sv_position = calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

            xs = sv_position[['x_sat', 'y_sat', 'z_sat']].to_numpy()
            ts = sv_position[['t_k']].to_numpy()
            pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
            pr = pr.to_numpy()
            if len(ecef_list) == 0:
                x, b, dp = least_squares_theres(xs, pr, x0, b0)
                # x, b, dp, av = earth_location1(xs, pr, x0)
                # x, b, dp, av = earth_location2(xs, pr, x0)
                # x, b, dp, av = least_squares3(xs, pr, x0, b0)
                # x, b, dp,av = least_squares1(xs, pr, x0, b0)
                # x, b, dp,av = least_squares2(xs, pr, x0, b0,ts)
            else:
                x, b, dp = least_squares_theres(xs, pr, x0, b0)
                # x, b, dp, av = earth_location1(xs, pr, x0)
                # x, b, dp, av = earth_location2(xs, pr, x0)
                # x, b, dp, av = least_squares3(xs, pr, x0, b0)
                # x, b, dp,av = least_squares1(xs, pr, ecef_list[-1], b0)
                # x, b, dp,av = least_squares2(xs, pr, ecef_list[-1], b0,ts)
            # x0 = np.array([0, 0, 0])
            # x, b, dp, av = earth_location2(xs, pr, x0)
            # x, b, dp,av = least_squares2(xs, pr, x0, b0,ts)
            # x, b, dp, av = least_squares1(xs, pr, x0, b0)
            # x, b, dp, av = least_squares3(xs, pr, x0, b0)
            ecef_list.append(x)
            # final_average_dist_list.append(av)
            lon, lat, alt = transformer.transform(x[0], x[1], x[2], radians=False)
            # print(lat, lon, alt)
            for i in range(len(xs)):
                row = []
                row.append(sats[i])
                row.append(timestamp)
                row.append(xs[i][0])
                row.append(xs[i][1])
                row.append(xs[i][2])
                row.append(lat)
                row.append(lon)
                row.append(alt)
                # row.append(av)
                row.append(0) # no av
                ans.append(row)


    print(final_average_dist_list)
    pd_ans = pd.DataFrame(ans,
                          columns=['sat_name', 'time_stamp', 'sat_x', 'sat_y', 'sat_z', 'lat', 'lon', 'alt', 'rad'])
    short = pd_ans.drop_duplicates(['time_stamp'])

    kml = simplekml.Kml()
    for row, col in short.iterrows():
        kml.newpoint(name=str(row), coords=[(col['lon'], col['lat'])])
        # pc = polycircles.Polycircle(latitude=col['lat'],
        #                                     longitude=col['lon'],
        #                                     radius=col['rad'],
        #                                     number_of_vertices=60)
        #
        # kml.newpolygon(name = str(row), outerboundaryis = pc.to_kml())
    kml.save("walking_new_try3.kml")
    # kml.save("fixed_new_try1.kml")


def find_location(file, measurements):
    ans = []
    manager = EphemerisManager(file.split('.')[0])
    ecef_list = []
    final_average_dist_list = []
    b0 = 0
    x0 = np.array([0, 0, 0])
    for epoch in measurements['Epoch'].unique():
        # one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)]
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch)]
        one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')
        if len(one_epoch.index) > 4:
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
            sats = one_epoch.index.unique().tolist()
            ephemeris = manager.get_ephemeris(timestamp, sats)
            sv_position = calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

            xs = sv_position[['x_sat', 'y_sat', 'z_sat']].to_numpy()
            ts = sv_position[['t_k']].to_numpy()
            pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
            pr = pr.to_numpy()
            if len(ecef_list) == 0:
                x, b, dp, av = earth_location1(xs, pr, x0)
                # x, b, dp, av = least_squares3(xs, pr, x0, b0)
                # x, b, dp,av = least_squares1(xs, pr, x0, b0)
                # x, b, dp,av = least_squares2(xs, pr, x0, b0,ts)
            else:
                x, b, dp, av = earth_location1(xs, pr, x0)
                # x, b, dp, av = least_squares3(xs, pr, x0, b0)
                # x, b, dp,av = least_squares1(xs, pr, ecef_list[-1], b0)
                # x, b, dp,av = least_squares2(xs, pr, ecef_list[-1], b0,ts)
            # x, b, dp,av = least_squares2(xs, pr, x0, b0,ts)
            # x, b, dp, av = least_squares1(xs, pr, x0, b0)
            # x, b, dp, av = least_squares3(xs, pr, x0, b0)
            ecef_list.append(x)
            final_average_dist_list.append(av)
            lon, lat, alt = transformer.transform(x[0], x[1], x[2], radians=False)
            # print(lat, lon, alt)
            for i in range(len(xs)):
                row = []
                row.append(sats[i])
                row.append(timestamp)
                row.append(xs[i][0])
                row.append(xs[i][1])
                row.append(xs[i][2])
                row.append(lat)
                row.append(lon)
                row.append(alt)
                row.append(av)
                ans.append(row)

    # print(ecef_list)
    print(final_average_dist_list)
    pd_ans = pd.DataFrame(ans,
                          columns=['sat_name', 'time_stamp', 'sat_x', 'sat_y', 'sat_z', 'lat', 'lon', 'alt', 'rad'])
    short = pd_ans.drop_duplicates(['time_stamp'])
    # print(short.shape)
    kml = simplekml.Kml()
    for row, col in short.iterrows():
        kml.newpoint(name=str(row), coords=[(col['lon'], col['lat'])])
        # pc = polycircles.Polycircle(latitude=col['lat'],
        #                                     longitude=col['lon'],
        #                                     radius=col['rad'],
        #                                     number_of_vertices=60)
        #
        # kml.newpolygon(name = str(row), outerboundaryis = pc.to_kml())
    kml.save("a1.kml")


def main():
    f = 'walking/gnss_log_2024_04_13_19_52_00.txt'
    # f = 'Ariel/gnss_log_2024_05_07_11_01_10.txt'
    # f = 'fixed/gnss_log_2024_04_13_19_51_17.txt'
    data = get_measurements(f)
    # find_location(f, data)
    csv_file = find_sat_location(f, data)
    find_earth_location_all(csv_file)


if __name__ == "__main__":
    main()
