import { ReactNode } from "react";


import Workspace from "./Workspace"
import LeftPanel from "./LeftPanel";


export default function PrimePage():ReactNode {

    return (
        <div className="flex flex-col h-screen sm:flex-row flex-nowrap sm:w-screen items-start justify-start">
            <LeftPanel />
            <div id='workspaceWrapper' className="h-full sm:flex sm:flex-row sm:pl-64 sm:h-full bg-gray-100 w-screen sm:w-full overflow-hidden relative">
                <Workspace/>
            </div>

        </div>
    )
}




