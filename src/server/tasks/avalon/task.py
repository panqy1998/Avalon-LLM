import sys
import json
from copy import deepcopy
from typing import List, Tuple, Dict, Any

from src.server.task import Task, Session
from src.typings import TaskSampleExecutionResult, TaskOutput, SampleIndex, AgentOutputStatus, SampleStatus
from src.utils import ColorMessage

from .engine import *
from .task_scoring import *

from .prompts import *
from .agents.baseline_agents import *

from .wrapper import FakeSession, SessionWrapper
from .utils import verbalize_team_result, verbalize_mission_result

from .agents.llm_with_discussion import LLMAgentWithDiscussion

from src.typings import AgentContextLimitException
from .avalon_exception import AvalonAgentActionException

from multi_agent.proxy import MultiAgentProxy

AGENT_FINDER = {
    'naive': find_naive_agent,
    'llm': LLMAgentWithDiscussion
}


class AvalonBench(Task):
    def __init__(self, num_players, agent_list, discussion, data_file, prompt="COT", **configs):
        super().__init__(**configs)

        self.num_players = num_players
        self.agent_list = agent_list

        self.discussion = discussion
        self.data_file = data_file
        self.prompt = prompt

        self.data: List[Tuple[dict, set]] = []
        with open(self.data_file, "r") as f:
            data_object = json.load(f)
        for data_item in data_object:
            self.data.append((data_item, -1))
        self.inputs = data_object

        self.seed = configs.pop('seed', 0)

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        outputs = [None for _ in range(len(self.data))]
        for result in results:
            outputs[result.index] = result.result

        win_counter = 0
        deduc_acc = 0
        valid_games = 0
        for result in results:
            if result.status == SampleStatus.COMPLETED:
                llm_idx = result.result['llm_idx']
                if result.result[f'Player_{llm_idx}_wins']:
                    win_counter += 1
                deduc_acc += result.result[f'Player_{llm_idx}_deduc_acc']
                valid_games += 1

        return {
            "Win rate of Player 0": win_counter / len(outputs),
            "Avg deduction acc of Player 0": deduc_acc / len(outputs),
        }

    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.data)))

    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        assert isinstance(index, int), "Index must be an integer"
        assert self.inputs[index]['num_players'] == self.num_players, "Number of players must be the same"
        proxy = MultiAgentProxy(session, self.num_players)
        sessions = [SessionWrapper(FakeSession(), proxy) for _ in range(self.num_players)]
        proxy.initialize_sessions(sessions)
        env = AvalonGameEnvironment.from_presets(self.inputs[index])
        scoring = AvalonScoring(env.config)

        true_player_sides = []
        believed_player_sides = []
        game_env_log = []

        num_players = self.num_players

        player_list = []

        if num_players != len(sessions):
            raise ValueError(
                f"Number of players {num_players} doesn't match number of sessions {len(sessions)}"
            )

        # Initialize players. Please remember to let Merlin and Evil players see the sides of all players.
        for i, (role_i, role_name, side) in enumerate(env.get_roles()):
            player_list.append(AGENT_FINDER[self.agent_list[i]](
                id=i,
                name=f"Player {i}",
                config=env.config,
                side=side,
                role=role_i,
                num_players=num_players,
                session=sessions[i],
                role_name=role_name,
                merlin=env.config.merlin,
                percival=env.config.percival,
                morgana=env.config.morgana,
                mordred=env.config.mordred,
                oberon=env.config.oberon,
                num_good=env.config.num_good,
                num_evil=env.config.num_evil,
                discussion=self.discussion,
                prompt=self.prompt,
                sides=env.get_partial_sides(i),
                seed=self.seed  # TODO: seed
            ))
            # If the player is Merlin or Evil, let them see the sides of all players.
            player_sides = [side for _, _, side in env.get_roles()]
            if player_list[i].role == 0 or player_list[i].side == 0:
                player_list[i].see_sides(player_sides)
                await player_list[i].initialize_game_info(player_list=env.get_roles())
            else:
                await player_list[i].initialize_game_info(player_list=[])

            proxy.get_next_agent()

        try:
            while not env.done:
                phase = env.get_phase()[0]
                print()
                print(ColorMessage.orange(f"##### Mission {env.turn}, Round {env.round} #####"))
                discussion_history = []
                # if phase is team selection phase, ask for team
                if phase == 0:
                    leader = env.get_quest_leader()
                    game_env_log.append(f"Selection Phase, the leader is Player {leader}")
                    print()
                    print(ColorMessage.cyan(f"##### System #####"))
                    print()
                    print(f"Selection Phase, the leader is Player {leader}")
                    """
                    Leader speaks & Discussion
                    """
                    discussion_history = []
                    if self.discussion:
                        # Leader speaks
                        print(f"Discussion")
                        proxy.set_current_agent(leader)
                        team, statement = await player_list[leader].team_discussion(
                            team_size=env.get_team_size(),
                            team=None,
                            team_leader_id=leader,
                            discussion_history=discussion_history,
                            mission_id=env.turn + env.round
                        )
                        discussion_history.append(f"Leader {leader} : " + statement + '\n')
                        print()
                        print(ColorMessage.cyan(f"##### LLM Agent (Player {leader}) #####"))
                        print()
                        print(ColorMessage.blue(f"Said: {statement}"))

                        # Discussion (sequential, once, in order for now) and Summarize
                        for idx, player in enumerate(player_list):
                            proxy.set_current_agent(idx)
                            if idx == leader:
                                continue
                            print()
                            print(ColorMessage.cyan(f"##### LLM Agent (Player {idx}) #####"))
                            print()
                            discussion = await player.team_discussion(
                                team_size=env.get_team_size(),
                                team=team,
                                team_leader_id=leader,
                                discussion_history=discussion_history,
                                mission_id=env.turn + env.round
                            )
                            discussion_history.append(f"Player {idx} : " + discussion + '\n')
                            print(ColorMessage.blue(f"{discussion}"))

                        for idx, player in enumerate(player_list):
                            proxy.set_current_agent(idx)
                            if self.agent_list[idx] == "llm":
                                await player.discussion_end(
                                    leader=leader,
                                    leader_statement=statement,
                                    discussion_history=discussion_history
                                )

                    # Choose a team
                    proxy.set_current_agent(leader)
                    if self.agent_list[leader] != "naive" or not self.discussion:
                        team = await player_list[leader].propose_team(
                            team_size=env.get_team_size(),
                            mission_id=env.turn,
                            discussion_history=discussion_history
                        )
                    env.choose_quest_team(
                        team=frozenset(team),
                        leader=leader
                    )
                    game_env_log.append(f"Leader Player {leader} chooses team {list(team)}")
                    print()
                    print(ColorMessage.cyan(f"##### System #####"))
                    print()
                    print(f"Leader Player {leader} chooses team {list(team)}")

                # if phase is team voting phase, ask for votes
                elif phase == 1:
                    game_env_log.append("Team Voting Phase")
                    print()
                    print(ColorMessage.cyan(f"##### System #####"))
                    print()
                    print("Team voting Phase")
                    votes = []
                    for i in range(num_players):
                        proxy.set_current_agent(i)
                        vote = await player_list[i].vote_on_team(
                            team=frozenset(env.get_current_quest_team()),
                            mission_id=env.turn,
                            discussion_history=discussion_history
                        )
                        votes.append(vote)
                        print(ColorMessage.cyan(f"Player {i} votes {vote}."))

                    # votes = [
                    #     await player_list[i].vote_on_team(
                    #         team                =   env.get_current_quest_team(),
                    #         mission_id          =   env.turn
                    #         ) for i in range(num_players)
                    #         ]
                    outcome = env.gather_team_votes(votes)
                    game_env_log.append(f"Team votes at this round: {str(votes)}")

                    # Observe results of Team Selection
                    for idx, player in enumerate(player_list):
                        proxy.set_current_agent(idx)
                        await player.observe_team_result(
                            mission_id=env.turn,
                            team=frozenset(env.get_current_quest_team()),
                            votes=votes,
                            outcome=outcome[2],
                        )

                    game_env_log.append(
                        "Team result: " + verbalize_team_result(team=env.get_current_quest_team(), votes=votes,
                                                                outcome=outcome[2]))
                    print()
                    print(ColorMessage.cyan(f"##### System #####"))
                    print()
                    print("Team result: " + verbalize_team_result(team=env.get_current_quest_team(), votes=votes,
                                                                  outcome=outcome[2]))


                # if phase is quest voting phase, ask for votes
                elif phase == 2:
                    game_env_log.append("Quest Voting Phase")
                    print()
                    print(ColorMessage.cyan(f"##### System #####"))
                    print()
                    print("Quest Voting Phase")
                    '''
                    TODO: Can have a discussion before voting on quest
                    '''
                    discussion_history = []
                    votes = []
                    for i in env.get_current_quest_team():
                        proxy.set_current_agent(i)
                        vote = await player_list[i].vote_on_mission(
                            team=frozenset(env.get_current_quest_team()),
                            mission_id=env.turn,
                            discussion_history=discussion_history
                        )
                        votes.append(vote)
                    # votes = [
                    #     await player_list[i].vote_on_mission(
                    #         team=env.get_current_quest_team(),
                    #         mission_id=env.turn,
                    #         discussion_history=discussion_history,
                    #         ) for i in env.get_current_quest_team()
                    #         ]
                    outcome = env.gather_quest_votes(votes)
                    game_env_log.append(f"Quest votes at this round: {str(votes)}")

                    # Observe mission/quest result
                    proxy.set_current_agent(0)
                    for idx, player in enumerate(player_list):
                        proxy.set_current_agent(idx)
                        await player.observe_mission(
                            team=frozenset(env.get_current_quest_team()),
                            mission_id=env.turn - 1,
                            num_fails=outcome[3],
                            votes=votes,
                            outcome=outcome[2],
                        )

                    game_env_log.append("Quest result: " + verbalize_mission_result(team=env.get_current_quest_team(),
                                                                                    outcome=outcome[2]))
                    print()
                    print(ColorMessage.cyan(f"##### System #####"))
                    print()
                    print("Quest result: " + verbalize_mission_result(team=env.get_current_quest_team(),
                                                                      outcome=outcome[2]))

                    # reflect sides of each player at the end of the game
                    for idx, player in enumerate(player_list):
                        proxy.set_current_agent(idx)
                        if self.agent_list[idx] == "llm":
                            believed_sides = await player.get_believed_sides(num_players=self.num_players)
                            game_env_log.append("Believed sides")
                            game_env_log.append(believed_sides)


                # if phase is assassination phase, ask for assassination
                elif phase == 3:
                    game_env_log.append("Assassination phase")
                    print()
                    print(ColorMessage.cyan(f"##### System #####"))
                    print()
                    print("Assassination phase")
                    '''
                        TODO: Discussion before Assassination Phase
                    '''
                    assassin = env.get_assassin()
                    proxy.set_current_agent(assassin)
                    target = int(await player_list[assassin].assassinate())

                    _, _, assassinated = env.choose_assassination_target(assassin, target)
                    game_env_log.append(f"Assassin Player {assassin} chooses to assassinate Player {target}")
                    print()
                    print(ColorMessage.cyan(f"##### System #####"))
                    print()
                    print(f"Assassin Player {assassin} chooses to assassinate Player {target}")
            # reflect sides of each player at the end of the game
            for idx, player in enumerate(player_list):
                proxy.set_current_agent(idx)
                if self.agent_list[idx] == "llm":
                    llm_believed_player_sides = await player.get_believed_sides(num_players=self.num_players)

                    true_player_sides.append(list(map(int, env.is_good)))
                    believed_player_sides.append(llm_believed_player_sides)

            if env.good_victory:
                answer = 1
            else:
                if sum(env.quest_results) >= 3:
                    answer = 0
                else:
                    answer = -1
            finish_reason = SampleStatus.COMPLETED
        except AgentContextLimitException as e1:
            chat_log = []
            for i in range(self.num_players):
                if self.agent_list[i] == "llm":
                    chat_log.append(player_list[i].session.log)
                else:
                    chat_log.append(None)
            result = {
                "game_env_log": game_env_log,
                "chat-log": chat_log, "error": e1
            }
            return TaskSampleExecutionResult(status=SampleStatus.AGENT_CONTEXT_LIMIT,
                                             result=result)
        except AvalonAgentActionException as e2:
            chat_log = []
            for i in range(self.num_players):
                if self.agent_list[i] == "llm":
                    chat_log.append(player_list[i].session.log)
                else:
                    chat_log.append(None)
            result = {
                "game_env_log": game_env_log,
                "chat-log": chat_log,
                "error": e2
            }
            return TaskSampleExecutionResult(status=SampleStatus.AGENT_INVALID_ACTION,
                                             result=result)
        except Exception as e:
            finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
            chat_log = []
            for i in range(self.num_players):
                if self.agent_list[i] == "llm":
                    chat_log.append(player_list[i].session.log)
                else:
                    chat_log.append(None)
            result = {
                "game_env_log": game_env_log,
                "chat-log": chat_log, "error": e
            }
            return TaskSampleExecutionResult(status=finish_reason, result=result)

        verbal_game_result = {
            -1: "Evil wins by mission!",
            0: "Evil wins by assassination!",
            1: "Good wins!"
        }
        chat_log = []
        for i in range(self.num_players):
            if self.agent_list[i] == "llm":
                chat_log.append(player_list[i].session.log)
            else:
                chat_log.append(None)
        llm_idx = [agent == "llm" for agent in self.agent_list]
        result = {"game_result": verbal_game_result[answer],
                  "llm_idx": llm_idx, "game_env_log": game_env_log,
                  "chat-log": chat_log}
        for id in llm_idx:
            result[f"role_of_Player_{id}"] = player_list[id].role_name
            result[f"Player_{id}_wins"] = (answer > 0) == bool(player_list[id].side)
        return TaskSampleExecutionResult(status=finish_reason, result=result)
