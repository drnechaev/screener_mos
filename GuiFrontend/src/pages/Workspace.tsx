import { createContext, ReactNode, useState, useCallback, useMemo, useEffect } from "react";
import { Routes, Route, Navigate, useLocation } from "react-router-dom";
import LoginPage, { LogOut } from "../auth/login";
import StudiesPage from "./StudiesPage";
import ImageDash from "./ImageDash";
import UploadPage from "./UploadStudy";

export type IImageContent = {
    study_name: string,
    series_id: number,
}

export type IShowImWorkspaceContext = {
    setShow: (content: IImageContent) => void
}

export const showImWorkpaceContext = createContext<IShowImWorkspaceContext | null>(null)

const Workspace = () => {
    const [isWorkspaceVisible, setIsWorkspaceVisible] = useState(false);
    const [imageContent, setImageContent] = useState<IImageContent>({ study_name: "", series_id: 0 });
    const location = useLocation();

    // Показываем workspace с контентом
    const setShowDashWorspace = useCallback((content: IImageContent) => {
        console.log(content)
        setImageContent(content);
        if (content.study_name && content.series_id)
            setIsWorkspaceVisible(true);
    }, []);

    // Закрываем workspace
    const closeWorkspace = useCallback(() => {
        setIsWorkspaceVisible(false);
    }, []);

    // Сбрасываем состояние при смене роута
    useEffect(() => {
        setIsWorkspaceVisible(false);
    }, [location.pathname]);

    const contextValue = useMemo(() => ({
        setShow: setShowDashWorspace
    }), [setShowDashWorspace]);

    return (
        <showImWorkpaceContext.Provider value={contextValue}>
            <div className="flex flex-1 p-3 h-full bg-blue-100 relative w-full">
                {/* Основной контент */}
                <div className={`w-full h-full overflow-y-auto transition-all duration-300 ${
                    isWorkspaceVisible ? 'sm:mr-[-33%] lg:mr-[-25%]' : ''
                }`}>
                    <Routes>
                        <Route path="*" element={<Navigate to="/Studies" />} />
                        <Route path='/login' element={<LoginPage />} />
                        <Route path='/Studies' element={<StudiesPage />} />
                        <Route path='/Upload' element={<UploadPage />} />
                        <Route path='/logout' element={<LogOut />} />
                    </Routes>
                </div>

                {/* Выезжающая панель ImageDash справа */}
                <div className={`
                    fixed top-0 right-0 h-full bg-white shadow-2xl z-50
                    transition-transform duration-300 ease-in-out
                    ${isWorkspaceVisible ? 'translate-x-0' : 'translate-x-full'}
                    w-full 
                    ${!isWorkspaceVisible ? 'pointer-events-none' : ''}
                `}>
                    {/* Кнопка закрытия внутри панели */}
                    <div className="absolute top-4 left-4 z-10">
                        <button
                            onClick={closeWorkspace}
                            className="p-2 bg-white rounded-full shadow-lg hover:bg-gray-100 transition-colors group"
                            title="Закрыть панель"
                        >
                            <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                    <ImageDash imageContent={imageContent} />
                </div>

                {/* Overlay для закрытия по клику вне панели */}
                <div className={`
                    fixed inset-0 bg-black z-40 sm:hidden
                    transition-opacity duration-300
                    ${isWorkspaceVisible ? 'opacity-50' : 'opacity-0 pointer-events-none'}
                `}
                    onClick={closeWorkspace}
                />
            </div>
        </showImWorkpaceContext.Provider>
    );
}

export default Workspace;