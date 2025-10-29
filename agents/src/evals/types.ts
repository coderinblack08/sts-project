export interface TestCase {
  "Attacker Tools": string[];
  Modifed: number;
  "Attacker Instruction": string;
  "Expected Achievements": string;
  "Attack Type": string;
  "User Tool": string;
  "User Instruction": string;
  "Tool Parameters": string;
  "Tool Response Template": string;
  Thought: string;
  "Tool Response": string;
}

export interface Toolkit {
  toolkit: string;
  category: string;
  name_for_model: string;
  name_for_human: string;
  description_for_model: string;
  description_for_human: string;
  tools: Array<{
    name: string;
    summary: string;
    parameters: Array<{
      name: string;
      type: string;
      description: string;
      required: boolean;
    }>;
    returns: Array<{
      name: string;
      type: string;
      description: string;
    }>;
    exceptions: Array<{
      name: string;
      description: string;
    }>;
  }>;
}
