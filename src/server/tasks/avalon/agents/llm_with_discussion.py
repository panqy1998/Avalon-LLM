from typing import List, Dict, Tuple
from .agent import Agent
from ..engine import AvalonBasicConfig
from ..wrapper import SessionWrapper, Session
from ..prompts import *
from copy import deepcopy
from ..utils import verbalize_team_result, verbalize_mission_result, get_team_result
from src.utils import ColorMessage


class LLMAgentWithDiscussion(Agent):
    r"""LLM agent with the ability to discuss with other agents."""

    def __init__(self, name: str, num_players: int, id: int, role: int, role_name: str, config: AvalonBasicConfig,
                 session: SessionWrapper = None, side=None, seed=None, **kwargs):
        self.name = name
        self.id = id
        self.num_players = num_players
        self.role = role
        self.role_name = role_name
        self.side = side  # 1 for good, 0 for evil
        self.session = session
        self.discussion = kwargs.pop('discussion', None)
        self.prompt = kwargs.pop('prompt', 'COT')
        self.infer_relation = [0.5] * num_players
        self.infer_relation[id] = 1
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.seed = seed

        self.config = config

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def see_sides(self, sides):
        self.player_sides = sides
        if self.side == 1:
            self.infer_relation = sides
        else:
            self.infer_relation = [1 - side for side in sides]

    async def initialize_game_info(self, player_list) -> None:
        """Initiliaze the game info for the agent, which includes game introduction, role, and reveal information for different roles."""
        # Introduction Prompt
        verbal_side = ["Evil", "Good"]
        intro_prompt = INTRODUCTION
        intro_prompt += '\n'
        content_prompt = intro_prompt + INFO_ROLE.format(self.num_players, self.num_good, int(self.merlin),
                                                         self.num_good - int(self.merlin) - int(self.percival),
                                                         self.num_evil,
                                                         self.num_evil - int(self.morgana) - int(self.mordred) - int(
                                                             self.oberon) - 1)
        identity_prompt = INFO_YOUR_ROLE.format(self.name, self.role_name, verbal_side[
            self.side])  # and do not pretend to be other roles throughout the game."
        self.identity_prompt = identity_prompt

        # Reveal Prompt
        reveal_info = ''
        minion_list = []
        servant_list = []
        assassin = ''
        merlin = ''
        for idx, player_info in enumerate(player_list):
            if player_info[1] == "Minion":
                minion_list.append(str(idx))
            elif player_info[1] == "Servant":
                servant_list.append(str(idx))
            elif player_info[1] == "Assassin":
                assassin = str(idx)
            elif player_info[1] == "Merlin":
                merlin = str(idx)
        if self.role_name == "Merlin":
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Merlin'][0].format(', '.join(minion_list), ', '.join(servant_list))
            elif len(minion_list) > 1:
                reveal_info = REVEAL_PROMPTS['Merlin'][1].format(', '.join(minion_list))
        if self.role_name == "Minion":
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Minion'][0].format(assassin, ', '.join(servant_list + [merlin]))
            elif len(minion_list) > 1:
                reveal_info = REVEAL_PROMPTS['Minion'][1].format(', '.join(minion_list))
        if self.role_name == "Assassin":
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Assassin'][0].format(', '.join(minion_list),
                                                                   ', '.join(servant_list + [merlin]))

        # Seperately pass the reveal info to the agent, so as to meet the requirement in filer_messages
        # TODO: is `system` allowed? 
        self.session.inject({
            "role": "user",
            "content": content_prompt,
            "mode": "system",
        })
        self.session.inject({
            # "role": "system",
            "role": "user",
            "content": identity_prompt + '\n' + reveal_info,
            "mode": "system",
        })

    async def summarize(self, mission_id=None, round_id=None) -> None:
        side = "good" if self.side == 1 else "evil"
        if mission_id is None:
            content = "You are Player {} with identity {} on the {} side." \
                      "Please summarize the history.  " \
                      "Try to keep all useful information,  including your identity, other player's identities, " \
                      "and your observations in the game.".format(self.id, self.role_name, side)
        else:
            content = "You are Player {} with identity {} on the {} side. Now you are at Mission {}, Round {}. " \
                      "Please summarize the history.  " \
                      "Try to keep all useful information,  including your identity, other player's identities, " \
                      "and your observations in the game.".format(self.id, self.role_name, side, mission_id, round_id)
        if self.prompt == "RELATION" or self.prompt == "RELATION+ICL":
            thought = RELATION_PROMPT.format(*self.infer_relation)
            content = content + "\n" + thought
        summary = await self.session.action({
            "role": "user",
            "content": content,
            "mode": "summarize"
        })
        # print("Summary: ", summary)
        past_history = deepcopy(self.session.get_history())
        self.session.overwrite_history(past_history[:2])
        self.session.inject({
            'role': "user",
            'content': summary
        })
        # print("History after summarization: ", self.session.get_history())

    async def observe_mission(self, team, mission_id, num_fails, votes, outcome, **kwargs) -> None:
        await self.session.action(
            {"role": "user",
             "content": verbalize_mission_result(team, outcome)}
        )

    async def observe_team_result(self, mission_id, team: frozenset, votes: List[int], outcome: bool) -> None:
        await self.session.action({
            "role": "user",
            "content": verbalize_team_result(team, votes, outcome),
        })

    async def get_believed_sides(self, num_players: int) -> List[float]:
        await self.summarize()
        input = {
            "role": "user",
            "content": GET_BELIEVED_SIDES.format(self.id),
            "mode": "get_believed_sides",
        }
        believed_player_sides = await self.session.action(input)
        print(f"Player {self.id} thinks:\n {believed_player_sides}")
        sides = []
        for p in range(self.config.num_players):
            input["target"] = p
            side = await self.session.parse_result(
                input=input,
                result=believed_player_sides
            )
            sides.append(side)
        print("Sides: ", sides)
        self.infer_relation = sides
        return sides

    async def discussion_end(self, leader: str, leader_statement: str, discussion_history: List[str]):
        content_prompt = (f"Discussion has ended. "
                          f"Here are the contents:\nStatement from Leader {leader}: {leader_statement}"
                          f"And words from other players:\n{' '.join(discussion_history)}")
        self.session.inject({
            "role": "user",
            "content": content_prompt,
        })
        await self.summarize()

    async def team_discussion(self, team_size, team, team_leader_id, discussion_history, mission_id, round_id):
        """Team discussion phase.

        We also summarize the history before this phase at each round. If there's no discussion phase, we summarize the history before the vote phase.
        """
        await self.summarize(mission_id, round_id)

        fails_required = self.config.num_fails_for_quest[mission_id]
        if self.id == team_leader_id:
            content_prompt = CHOOSE_TEAM_ACTION.format(team_size, self.num_players - 1) + " " + CHOOSE_TEAM_LEADER
            input = {"content": content_prompt, "mode": "choose_quest_team_action with discussion", "role": "user",
                     "team": team, "team_leader_id": team_leader_id, "team_size": team_size}
            statement = await self.session.action(input)
            proposed_team = await self.session.parse_result(input, statement)
            return proposed_team, statement
        else:
            content_prompt = ' '.join(discussion_history) + ' ' + VOTE_TEAM_DISCUSSION.format(self.id, list(team))
            input = {"content": content_prompt, "role": "user"}
            discussion = await self.session.action(input)
            return discussion

    async def quest_discussion(self, team_size, team, team_leader_id, discussion_history, mission_id):
        fails_required = self.config.num_fails_for_quest[mission_id]

    async def propose_team(self, team_size, mission_id, discussion_history):
        if not discussion_history:
            await self.summarize()
        content_prompt = CHOOSE_TEAM_ACTION.format(team_size, self.num_players - 1)
        if self.prompt == "COT":
            thought = COTHOUGHT_PROMPT
        elif self.prompt == "RELATION":
            thought = RELATION_PROMPT.format(*self.infer_relation)
        elif self.prompt == "ICL":
            if self.role_name == "minion":
                thought = ICL_PROPOSE_TEAM_MINION_PROMPT
            elif self.role_name == "assasin":
                thought = ICL_PROPOSE_TEAM_ASSASIN_PROMPT
            else:
                thought = ICL_PROPOSE_TEAM_PROMPT
        elif self.prompt == "RELATION+ICL":
            thought = RELATION_PROMPT.format(*self.infer_relation)
            if self.role_name == "minion":
                thought += ICL_PROPOSE_TEAM_MINION_PROMPT
            elif self.role_name == "assasin":
                thought += ICL_PROPOSE_TEAM_ASSASIN_PROMPT
            else:
                thought += ICL_PROPOSE_TEAM_PROMPT

        else:
            thought = ""
        input = {
            "role": "user",
            "content": content_prompt + '\n' + thought,
            "team_size": team_size,
            "seed": self.seed,
            "role_name": self.role_name,
            "mode": "choose_quest_team_action",
        }
        proposed_team = await self.session.action(input)

        print()
        print(ColorMessage.cyan(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####"))
        print()
        print(ColorMessage.blue(f"Thought: {proposed_team}"))

        if proposed_team[0] == "[" and proposed_team[-1] == "]":
            proposed_team2 = get_team_result(proposed_team)
            if proposed_team2 != team_size:
                proposed_team2 = await self.session.parse_result(input, proposed_team)
            proposed_team = proposed_team2
        elif isinstance(self.session.session, Session):
            proposed_team = await self.session.parse_result(input, proposed_team)
        proposed_team = frozenset(proposed_team)

        if isinstance(proposed_team, frozenset):
            return proposed_team
        else:
            raise ValueError(
                "Type of proposed_team must be frozenset, instead of {}.".format(type(proposed_team))
            )

    async def vote_on_team(self, team, mission_id, discussion_history):
        """Vote to approve or reject a team.

        If there's no discussion phase, we will summarize the history before the vote phase.
        """
        if not discussion_history:
            await self.summarize()

        content_prompt = VOTE_TEAM_ACTION.format(self.id, list(team))
        if self.prompt == "COT":
            thought = COTHOUGHT_PROMPT
        elif self.prompt == "RELATION":
            thought = RELATION_PROMPT.format(*self.infer_relation)
        elif self.prompt == "ICL":
            if self.role == "servant":
                thought = ICL_VOTE_TEAM_SERVANT_PROMPT
            elif self.role == "merlin":
                thought = ICL_VOTE_TEAM_MERLIN_PROMPT
            else:
                thought = ""
        elif self.prompt == "RELATION+ICL":
            thought = RELATION_PROMPT.format(*self.infer_relation) + "\n"
            if self.role == "servant":
                thought += ICL_VOTE_TEAM_SERVANT_PROMPT
            elif self.role == "merlin":
                thought += ICL_VOTE_TEAM_MERLIN_PROMPT
        else:
            thought = ""
        input = {
            "role": "user",
            "content": content_prompt + "\n" + thought,
            "side": int(self.side),
            "mode": "vote_on_team",
            "seed": self.seed,
            "role_name": self.role_name,
        }
        vote_result = await self.session.action(input)

        print()
        print(ColorMessage.cyan(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####"))
        print()
        print(ColorMessage.blue(f"Thought: {vote_result}"))
        if vote_result not in ["Yes", "No"]:
            if isinstance(self.session.session, Session):
                vote_result = await self.session.parse_result(input, vote_result)
        else:
            vote_result = 1 if vote_result == "Yes" else 0
            verbal_team_act = {
                0: "Reject the team",
                1: "Approve the team"
            }
            print(ColorMessage.blue("Action:") + " ", verbal_team_act[vote_result])

        if isinstance(vote_result, int):
            return vote_result
        else:
            raise ValueError(
                "Vote result should be either 0 or 1, instead of {}.".format(type(vote_result))
            )

    async def vote_on_mission(self, team, mission_id, discussion_history):
        await self.summarize()
        content_prompt = VOTE_MISSION_ACTION.format(list(team))
        if self.prompt == "COT":
            thought = COTHOUGHT_PROMPT
        elif self.prompt == "RELATION" or self.prompt == "RELATION+ICL":
            thought = RELATION_PROMPT.format(*self.infer_relation)
        else:
            thought = ""
        input = {
            "role": "user",
            "content": content_prompt + "\n" + thought,
            "side": int(self.side),
            "mode": "vote_on_mission",
            "seed": self.seed,
            "role_name": self.role_name,
        }
        vote_result = await self.session.action(input)

        print()
        print(ColorMessage.cyan(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####"))
        print()
        print(ColorMessage.blue(f"Thought: {vote_result}"))
        if vote_result not in ["Yes", "No"]:
            if isinstance(self.session.session, Session):
                vote_result = await self.session.parse_result(input, vote_result)
        else:
            vote_result = 1 if vote_result == "Yes" else 0
        if isinstance(vote_result, int):
            return vote_result
        else:
            raise ValueError(
                "Vote result should be either 0 or 1, instead of {}.".format(type(vote_result))
            )

    async def assassinate(self):
        if self.role != 7:
            raise ValueError("Only the Assassin can assassinate.")
        await self.summarize()
        if self.prompt == "COT":
            thought = COTHOUGHT_PROMPT
        elif self.prompt == "RELATION" or self.prompt == "RELATION+ICL":
            thought = RELATION_PROMPT.format(*self.infer_relation)
        input = {
            "role": "user",
            "content": ASSASSINATION_PHASE.format(self.num_players - 1) + "\n" + thought,
            "mode": "assassination",
            "seed": self.seed,
            "role_name": self.role_name,
        }
        assassinate_result = await self.session.action(input)

        print()
        print(ColorMessage.cyan(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####"))
        print()
        print(ColorMessage.blue(f"Thought: {assassinate_result}"))
        try:
            assassinate_result = int(assassinate_result)
        except ValueError:
            if isinstance(self.session.session, Session):
                assassinate_result = await self.session.parse_result(input, assassinate_result)

        if isinstance(assassinate_result, int):
            return assassinate_result
        else:
            raise ValueError(
                "Assassination result should be an integer, instead of {}.".format(type(assassinate_result))
            )
