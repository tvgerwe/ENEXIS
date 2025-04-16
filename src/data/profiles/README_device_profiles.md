# 📦 Device Usage Profiles — README

This file describes the structure and purpose of the `device_usage_profile.json` file, used for modeling household energy consumption in combination with time-based electricity price forecasts.

## 🔍 Purpose
The JSON file contains structured usage behavior per device, allowing accurate simulation and optimization of energy consumption based on:
- Realistic operational behavior (duration, frequency)
- Device-specific constraints (time windows, minimum runtime)
- Load shifting flexibility
- Integration with hourly energy price forecasts

## 🧱 Structure per Device

Each device entry includes:

| Field                  | Type         | Description                                                                 |
|------------------------|--------------|-----------------------------------------------------------------------------|
| `Device`               | string       | Name of the appliance                                                      |
| `Duration_h`           | float        | Duration of one usage cycle in hours                                       |
| `kWh_per_use`          | float        | Average energy consumption per use (in kWh)                                |
| `Weekly_Uses`          | integer      | Number of times the device is used per week                                |
| `Flexibility`          | string       | Describes how flexible the device is for shifting usage (`Yes`, `Partial`) |
| `Constraints`          | object       | See below                                                                  |
| `Source`               | string (URL) | Reference source supporting the usage pattern                              |

### 🔧 Constraints Object

The `Constraints` object defines behavior limitations for optimization:

| Field                   | Type    | Description                                                                 |
|-------------------------|---------|-----------------------------------------------------------------------------|
| `Allowed_Hours`         | int[]   | Hours of the day when the device is allowed to operate (0 = 00:00–01:00)   |
| `Min_Block_Hours`       | float   | Minimum continuous block of operation (in hours)                           |
| `Max_Skip_Hours`        | int     | Maximum allowed time between usage blocks                                  |
| `Max_Simultaneous_Use`  | int     | Max allowed simultaneous operation with other devices (for load balancing) |

## ⚙️ How to Use in Optimization

- Filter forecast data by `Allowed_Hours`
- Use block-based optimization to find cheapest available slots
- Respect `Weekly_Uses` and total energy needs
- Apply realistic behavior constraints such as `Max_Skip_Hours` (e.g., for heat pumps)
- Calculate potential savings by comparing against uncontrolled usage

## 📌 Example

```json
{
  "Device": "EV Charging",
  "Duration_h": 6.0,
  "kWh_per_use": 15.0,
  "Weekly_Uses": 3,
  "Flexibility": "Yes",
  "Constraints": {
    "Allowed_Hours": [22, 23, 0, 1, 2, 3, 4, 5, 6],
    "Min_Block_Hours": 2,
    "Max_Skip_Hours": 0,
    "Max_Simultaneous_Use": 1
  },
  "Source": "https://www.mdpi.com/1996-1073/17/4/925"
}


🔗 Sources
Washing Machine
"Best Time to Run Your Washing Machine to Save Money" — Better Homes & Gardens
https://www.bhg.com/the-best-time-to-run-your-washing-machine-to-save-money-11702075

Dishwasher
"How Much Power Does a Dishwasher Use Per Cycle?" — HomeGearGeek
https://homegeargeek.com/how-much-power-does-a-dishwasher-use-per-cycle

Heat Pump
"How Much Electricity Does a Heat Pump Use Per Day?" — Appliance Mastery
https://appliancemastery.com/how-much-electricity-does-a-heat-pump-use-per-day

EV Charging
"A Survey on Electric Vehicle Energy Consumption and Charging Patterns" — MDPI
https://www.mdpi.com/1996-1073/17/4/925