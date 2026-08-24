# Header standard con timestamp e frame
std_msgs/Header header

# Potenza istantanea totale stimata [W]
float64 instantaneous_power

# Potenza media su una finestra mobile [W] (riduzione rumore)
float64 average_power

# Energia cumulata dall'avvio del nodo [J]
float64 cumulative_energy

# Potenza stimata per singolo giunto [W], stesso ordine di /joint_states
string[] joint_names
float64[] joint_power
