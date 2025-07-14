import fastf1

session = fastf1.get_session(2021, 7, 'Q')

session.load()
fastest_lap = session.laps.pick_fastest()
print(fastest_lap['LapTime'])
print(fastest_lap['Driver'])