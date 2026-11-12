import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "PromptCondenser.Queue",
    setup() {
        api.addEventListener("prompt_condenser.queue", () => {
            app.queuePrompt(0, 1);
        });
    }
});
