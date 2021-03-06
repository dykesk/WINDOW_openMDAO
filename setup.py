from distutils.core import setup

setup(
    name='WINDOW_openMDAO',
    version='',
    description="WINDOW is an MDAO workflow to design offshore wind farms",
    long_description="""
    WINDOW is an MDAO workflow to design offshore wind farms. It is written in
    NASA's openMDAO framework. The following publication shows the Extended
    Design Structure Matrix of WINDOW_openMDAO:

    Sebastian Sanchez Perez Moreno and Michiel B. Zaaijer. "How to select MDAO workflows",
    2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference,
    AIAA SciTech Forum, (AIAA 2018-0654) https://doi.org/10.2514/6.2018-0654
    """,
    author="Sebastian Sanchez Perez-Moreno",
    author_email="s.sanchezperezmoreno@tudelft.nl",
    url='https://github.com/sebasanper/WINDOW_openMDAO',
    packages=[
      'WINDOW_openMDAO',
      'WINDOW_openMDAO.Costs',
      'WINDOW_openMDAO.Costs.costs',
      'WINDOW_openMDAO.Costs.costs.decommissioning_costs',
      'WINDOW_openMDAO.Costs.costs.investment_costs',
      'WINDOW_openMDAO.Costs.costs.investment_costs.installation_costs',
      'WINDOW_openMDAO.Costs.costs.investment_costs.procurement_costs',
      'WINDOW_openMDAO.Costs.costs.investment_costs.procurement_costs.RNA_costs',
      'WINDOW_openMDAO.Costs.costs.investment_costs.procurement_costs.auxiliary_costs',
      'WINDOW_openMDAO.Costs.costs.investment_costs.procurement_costs.electrical_system_costs',
      'WINDOW_openMDAO.ElectricalCollection',
      'WINDOW_openMDAO.Finance',
      'WINDOW_openMDAO.OandM',
      'WINDOW_openMDAO.SupportStructure',
      'WINDOW_openMDAO.SupportStructure.teamplay_folder',
      'WINDOW_openMDAO.SupportStructure.teamplay_folder.lib',
      'WINDOW_openMDAO.SupportStructure.teamplay_folder.lib.analysts_humanities',
      'WINDOW_openMDAO.SupportStructure.teamplay_folder.lib.analysts_physics',
      'WINDOW_openMDAO.SupportStructure.teamplay_folder.lib.designers_support',
      'WINDOW_openMDAO.SupportStructure.teamplay_folder.lib.environment',
      'WINDOW_openMDAO.SupportStructure.teamplay_folder.lib.system',
      'WINDOW_openMDAO.Turbine',
      'WINDOW_openMDAO.WakeModel',
      'WINDOW_openMDAO.WakeModel.Turbulence',
      'WINDOW_openMDAO.WakeModel.WakeMerge',
      'WINDOW_openMDAO.WaterDepth',
      'WINDOW_openMDAO.AEP',
      'WINDOW_openMDAO.AEP.FastAEP',
      'WINDOW_openMDAO.AEP.FastAEP.farm_energy',
      'WINDOW_openMDAO.AEP.FastAEP.farm_energy.wake_model_mean_new',
      'WINDOW_openMDAO.AEP.FastAEP.farm_energy.wake_model_mean_new.aero_power_ct_models',
      'WINDOW_openMDAO.AEP.FastAEP.site_conditions',
      'WINDOW_openMDAO.AEP.FastAEP.site_conditions.wind_conditions',
      'WINDOW_openMDAO.src',
      'WINDOW_openMDAO.src.AbsAEP',
      'WINDOW_openMDAO.src.AbsCosts',
      'WINDOW_openMDAO.src.AbsElectricalCollection',
      'WINDOW_openMDAO.src.AbsFinance',
      'WINDOW_openMDAO.src.AbsOandM',
      'WINDOW_openMDAO.src.AbsSupportStructure',
      'WINDOW_openMDAO.src.AbsTurbine',
      'WINDOW_openMDAO.src.AbsTurbulence',
      'WINDOW_openMDAO.src.AbsWakeModel',
      'WINDOW_openMDAO.src.AbsWakeModel.AbsWakeMerge',
      'WINDOW_openMDAO.src.SiteConditionsPrep',
      'WINDOW_openMDAO.src.Utils'
    ],
    package_data={'WINDOW_openMDAO': ['Input/*.dat', 'Input/*.pkl',
                                      'WakeModel/*.dat']}
)
