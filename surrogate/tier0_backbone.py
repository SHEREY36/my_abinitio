"""tier0_backbone.py  -- differentiable two-resistance backbone for RP-CVD Si epitaxy.
Import: from tier0_backbone import growth_rate, diagnostics, PARAMS
This is the low-fidelity model f_LF that Tier 1 calibrates against CFD-ACE+.
"""
import jax, jax.numpy as jnp
from jax import grad
jax.config.update("jax_enable_x64", True)

R_u=8.314462618; k_B=1.380649e-23; N_A=6.02214076e23; eV=1.602176634e-19
M_SiH4=32.117e-3; m_SiH4=M_SiH4/N_A; M_Si=28.0855e-3; rho_Si=2329.0
Omega_Si=M_Si/rho_Si; Gamma=6.78e18/N_A

PARAMS=dict(s0=1.0e-2, Ea_ads=0.0*eV, nu_des=1.0e13, Ea_des=2.0*eV, beta=2.0)

def k_ads_eff(T,P): return P["s0"]*jnp.exp(-P["Ea_ads"]/(k_B*T))/(jnp.sqrt(2*jnp.pi*m_SiH4*k_B*T)*N_A)
def k_des_eff(T,P): return P["nu_des"]*Gamma*jnp.exp(-P["Ea_des"]/(k_B*T))
def theta_H_steady(p_s,T_s,P):
    r=jnp.sqrt(P["beta"]*k_ads_eff(T_s,P)*p_s/k_des_eff(T_s,P)); return r/(1+r)
def surface_rate(p_s,T_s,P):
    thH=theta_H_steady(p_s,T_s,P); R=k_ads_eff(T_s,P)*p_s*(1-thH)**2
    return Omega_Si*R, R, thH
def diffusivity(T,p_tot,D_ref=6.0e-5,T_ref=300.0,p_ref=101325.0):
    return D_ref*(T/T_ref)**1.75*(p_ref/p_tot)
def k_m_stagnation(a,D,L):
    bl=jnp.sqrt(jnp.pi*D/(2*a)); return D/(bl*jax.scipy.special.erf(L*jnp.sqrt(a/(2*D))))
def coupled_solve(th):
    T_s=th["T_s"]; D=diffusivity(T_s,th["p_tot"]); km=k_m_stagnation(th["a_strain"],D,th["L"])
    c_inf=th["p_SiH4"]/(R_u*T_s)
    def resid(c):
        _,R,_=surface_rate(c*R_u*T_s,T_s,th["P"]); return km*(c_inf-c)-R
    def body(i,c): return c-resid(c)/grad(resid)(c)
    c=jax.lax.fori_loop(0,60,body,0.5*c_inf); return c*R_u*T_s,km,c_inf,D
def growth_rate(th):
    p_s,km,c_inf,D=coupled_solve(th); return surface_rate(p_s,th["T_s"],th["P"])[0]
def diagnostics(th):
    p_s,km,c_inf,D=coupled_solve(th); G,R,thH=surface_rate(p_s,th["T_s"],th["P"])
    c_s=p_s/(R_u*th["T_s"]); Da=float((R/c_s)/km)
    return dict(G_nmmin=float(G)*1e9*60, theta_H=float(thH), Da=Da,
                depletion=float(1-c_s/c_inf), k_m=float(km))
