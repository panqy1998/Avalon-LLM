from transformers import AutoTokenizer, AutoModelForCausalLM
from ..agent import AgentClient
from src.typings import *
from src.utils import *


class Prompter:
    @staticmethod
    def get_prompter(prompter: Union[Dict[str, Any], None]):
        # check if prompter_name is a method and its variable
        if not prompter:
            return Prompter.default()
        assert isinstance(prompter, dict)
        prompter_name = prompter.get("name", None)
        prompter_args = prompter.get("args", {})
        if hasattr(Prompter, prompter_name) and callable(
                getattr(Prompter, prompter_name)
        ):
            return getattr(Prompter, prompter_name)(**prompter_args)
        return Prompter.default()

    @staticmethod
    def default():
        return Prompter.role_content_dict()

    @staticmethod
    def batched_role_content_dict(*args, **kwargs):
        base = Prompter.role_content_dict(*args, **kwargs)

        def batched(messages):
            result = base(messages)
            return {key: [result[key]] for key in result}

        return batched

    @staticmethod
    def role_content_dict(
            message_key: str = "messages",
            role_key: str = "role",
            content_key: str = "content",
            user_role: str = "user",
            agent_role: str = "agent",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal message_key, role_key, content_key, user_role, agent_role
            role_dict = {
                "user": user_role,
                "agent": agent_role,
            }
            prompt = []
            for item in messages:
                prompt.append(
                    {role_key: role_dict[item["role"]], content_key: item["content"]}
                )
            return {message_key: prompt}

        return prompter

    @staticmethod
    def prompt_string(
            prefix: str = "",
            suffix: str = "AGENT:",
            user_format: str = "USER: {content}\n\n",
            agent_format: str = "AGENT: {content}\n\n",
            prompt_key: str = "prompt",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal prefix, suffix, user_format, agent_format, prompt_key
            prompt = prefix
            for item in messages:
                if item["role"] == "user":
                    prompt += user_format.format(content=item["content"])
                else:
                    prompt += agent_format.format(content=item["content"])
            prompt += suffix
            print(prompt)
            return {prompt_key: prompt}

        return prompter

    @staticmethod
    def claude():
        return Prompter.prompt_string(
            prefix="",
            suffix="Assistant:",
            user_format="Human: {content}\n\n",
            agent_format="Assistant: {content}\n\n",
        )

    @staticmethod
    def palm():
        def prompter(messages):
            return {"instances": [
                Prompter.role_content_dict("messages", "author", "content", "user", "bot")(messages)
            ]}

        return prompter


class TransformerAgent(AgentClient):
    def __init__(
            self,
            model_name="meta-llama/Llama-2-7b-chat-hf",
            access_token="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",
            load_in_4bit=True, cache_dir="src/client/agents/model/",
            use_fast=True,
            proxies=None,
            body=None,
            headers=None,
            return_format="{response}",
            prompter=None,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.access_token = access_token
        self.load_in_4bit = load_in_4bit
        self.cache_dir = cache_dir
        self.use_fast = use_fast
        self.body = body or {}
        self.return_format = return_format
        self.prompter = Prompter.get_prompter(prompter)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto",
                                                              load_in_4bit=self.load_in_4bit,
                                                              token=self.access_token,
                                                              cache_dir=self.cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=self.use_fast,
                                                           token=self.access_token)
        except:
            raise Exception("Cannot load {}".format(self.model_name))

    def _handle_history(self, history: List[dict]) -> Dict[str, Any]:
        return self.prompter(history)

    def inference(self, history: List[dict]) -> str:
        try:
            body = self.body.copy()
            body.update(self._handle_history(history))
            prompt = body["messages"]
            new_prompts = []
            for i, message in enumerate(prompt):
                if (message["role"] == "user") and (message["content"] != ""):
                    new_prompts.append(message["content"])
            new_prompt = [{"role": "user", "content": "\n".join(new_prompts)}]
            model_inputs = self.tokenizer.apply_chat_template(new_prompt, tokenize=True, add_generation_prompt=True,
                                                              return_tensors="pt")
            resp = self.model.generate(model_inputs.to('cuda'), max_new_tokens=512)
            resp = self.tokenizer.decode(resp[0])
            resp = resp[(resp.rfind("[/INST]") + len("[/INST]")):].replace("</s>", "")
        except AgentClientException as e:
            raise e
        except Exception as e:
            print("Warning: ", e)
            pass
        else:
            return self.return_format.format(response=resp)
        raise Exception("Failed.")
