import roleml
import cassiopeia as cass

cass.set_riot_api_key("RGAPI-")
cass.set_default_region("EUW")

summoner = cass.get_summoner(name="Canisback")
match = cass.MatchHistory(summoner=summoner, queues={cass.Queue.blind_fives})[0].load()
match.timeline.load()

roleml.add_cass_predicted_roles(match)
for p in match.participants:
    print(p.predicted_role)