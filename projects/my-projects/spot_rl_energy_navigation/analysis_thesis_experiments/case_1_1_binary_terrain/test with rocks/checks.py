"""Validate replay commands against the original source time series."""
import csv
import math
from references import HERE,rows


def validate_forward_commands(output,cases):
    references={c['mission']['mission_id']:rows(HERE/'inputs'/c['reference_file']) for c in cases}
    with (output/'trajectories.csv').open() as stream:
        for row in csv.DictReader(stream):
            step=int(row['step'])
            if not step:continue
            trace=references[row['mission_id']]
            index=min(step,len(trace)-1)
            if not math.isclose(float(row['command_forward_m_s']),float(trace[index]['command_forward_m_s']),abs_tol=1e-6):
                raise ValueError('Applied forward command differs from the source time series')
            if int(row['reference_step'])!=index:
                raise ValueError('Wrong reference step')
            if int(row['speed_profile_extended'])!=int(step>=len(trace)):
                raise ValueError('Wrong speed-profile extension flag')
