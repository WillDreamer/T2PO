from gym.envs.registration import register

from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs:WebAgentSiteEnv',
  kwargs={"disable_env_checker": True},
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
  kwargs={"disable_env_checker": True},
)