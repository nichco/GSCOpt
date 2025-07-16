# luca.py
# This file contains functions to calculate takeoff and landing distances
# based on the equations from Anderson's "Introduction to Flight".



def takeoff(W, # weight (N)
            T, # thrust (N)
            S, # wing area (m^2)
            rho=0.91, # air density (kg/m^3) 0.91 at 9000 ft
            g=9.81, # gravity (m/s^2)
            CL_max=1.5, # max lift coefficient
            ):
    
    # Anderson Intro to Flight equation 6.104
    
    S_LO = 1.44 * W**2 / (g * rho * S * CL_max * T)

    return S_LO



def landing(W, # weight (N)
            S, # wing area (m^2)
            V_T=70, # touchdown speed (m/s)
            CD_0=0.02, # zero-lift drag coefficient
            L=0, # lift (N) fair to assume lift is zero if spoilers are deployed
            rho=0.91, # air density (kg/m^3) 0.91 at 9000 ft
            g=9.81, # gravity (m/s^2)
            CL_max=1.5, # max lift coefficient
            mu_r=0.4, # runway friction coefficient with brakes
            ):
    
    # Anderson Intro to Flight equation 6.111
    
    D = 0.5 * rho * V_T**2 * S * CD_0 / 2

    denominator = g * rho * S * CL_max * (D + mu_r * (W - L))

    S_L = 1.69 * W**2 / denominator

    return S_L







if __name__ == "__main__":
    # approx. 737 numbers
    W = 90000 * 9.81 # N
    T = 133500 # N
    S = 130 # m^2

    S_lo = takeoff(W, T, S)
    S_lo_ft = S_lo * 3.28084 # convert to feet
    print(f"Takeoff distance: {S_lo:.2f} m")
    print(f"Takeoff distance: {S_lo_ft:.2f} ft")

    S_landing = landing(W, S)
    S_landing_ft = S_landing * 3.28084 # convert to feet
    print(f"Landing distance: {S_landing:.2f} m")
    print(f"Landing distance: {S_landing_ft:.2f} ft")