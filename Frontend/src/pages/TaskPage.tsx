import { TaskWorkbench } from "@/components/workbench/TaskWorkbench";
import { TaskDefinition } from "@/tasks/types";

interface TaskPageProps {
  task: TaskDefinition;
}

const TaskPage = ({ task }: TaskPageProps) => {
  // key forces a full workbench remount (state reset) when switching tasks
  return <TaskWorkbench key={task.id} task={task} />;
};

export default TaskPage;
