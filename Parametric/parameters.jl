# developed with julia 1.5.3
#
# PV unit physical values

using Distributed

@everywhere begin

	const peak_power = 1000. # kW
	const max_price = 0.6 # EUR/kWh
	const min_price = 0.4 # EUR/kWh
	const price = vcat(ones(38)*min_price, ones(4)*max_price, ones(6)*min_price)
	const penalty_coefficient = 2.

	const max_battery_capacity = 1000. # kWh
	const max_battery_power = 1000. # kW
	const dt = 0.5 # h 
	const horizon = 48
	const rho_c = 0.95
	const rho_d = 0.95

end