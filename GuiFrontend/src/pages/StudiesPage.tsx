import { Dispatch, ReactNode, useContext, useEffect, useState } from "react";
import { IShowImWorkspaceContext, showImWorkpaceContext } from "./Workspace";
import axios, { AxiosError } from "axios";
import { LogOutResponeCheck } from "../auth/auth";

type StudyInfo = {
    StudyID: string,
    Modalities: string,
    Time: string,
    NumStudies: string
    SetComments: boolean
}

async function fetchStudy(studyID: string): Promise<any> {
    try {
        const { data: response } = await axios.get(`${import.meta.env.VITE_GUI_BACKEND}/studies/${studyID}`); 
        return response;
    } catch (error) {
        console.log(error);
        return []
    }
}

async function getStudies(setStudies: Dispatch<Array<StudyInfo>>): Promise<any> {
    try {
        console.log("Send query for study")
        const response = await axios.get(`${import.meta.env.VITE_GUI_BACKEND}/studies`)
        if (response.data) {
            console.log("Getted studies data") 
            console.log(response.data.Studies)
            setStudies(response.data.Studies)
        }
    } catch (error: AxiosError | any) {
        LogOutResponeCheck(error)
    }  
}


const DownloadButton = ({filename}: {filename: string }) => {
    const handleDownload = async () => {
        try {
        const response = await axios.get(`${import.meta.env.VITE_GUI_BACKEND}/studies/download/`, {
            responseType: 'blob'
        });
        
        const blob = new Blob([response.data]);
        const downloadUrl = window.URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = filename;
        link.click();
        
        window.URL.revokeObjectURL(downloadUrl);
        } catch (error) {
        console.error('Download failed:', error);
        }
    };

    return (
        <button 
        onClick={handleDownload}
        className="px-4 py-2 my-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
        Скачать результаты в формате xlsx
        </button>
    );
};


const StudiesPage = () => {
    const { setShow } = useContext(showImWorkpaceContext) as IShowImWorkspaceContext
    const [studies, setStudies] = useState<Array<StudyInfo>>([])
    
    useEffect(() => {
        getStudies(setStudies)
        return () => setShow({ study_name: "", series_id: 0 })
    }, [])

    // Ставим интервал на проверку
    useEffect(() => {
        const interval = setInterval(() => {
            getStudies(setStudies)
        }, 30000);
        return () => clearInterval(interval);
    }, []);

    const showStudy = () => {
        console.log("SD")
            setShow()
    }

    return (
        <>
        <DownloadButton filename="output.xlsx" />
        <table className="w-full text-sm text-center text-gray-500 table-auto">
            <thead className="font-bold text-gray-700 uppercase bg-gray-50">
                <tr>
                    <th scope="col" className="px-6 py-3">
                        Исследование
                    </th>
                    <th scope="col" className="px-6 py-3">
                        Статус обработки
                    </th>
                    <th scope="col" className="px-6 py-3">
                        Наличие патологии
                    </th>
                    <th scope="col" className="px-6 py-3">
                        Где патология
                    </th>
                </tr>
            </thead>
            <tbody>
                {studies.map((key) => (
                    <tr 
                        key={key.series_uid + key.studyName} 
                        className={`${key.proccessed === false ? 'bg-green-300' : (key.processing_status === 'Success' ? 'bg-blue-300' : 'bg-red-300')} border-b text-center hover:*:underline cursor-pointer`}
                    >
                        <td 
                            scope="row" 
                            className="px-6 py-4 font-medium text-gray-900 break-all" 
                            onClick={key.series_uid ? () => setShow({series_id:key.series_uid, study_name:key.studyName}) : undefined}
                        >
                            {key.studyName} {key.study_uid && <div> {key.study_uid + "/" + key.series_uid}</div>} 
                        </td>
                        <td 
                            className="px-6 py-4" 
                            onClick={key.series_uid ? () => setShow({series_id:key.series_uid, study_name:key.studyName}) : undefined}
                        >
                            {key.proccessed !== false ? key.processing_status : "Pending"}
                        </td>
                        <td 
                            className="px-6 py-4" 
                            onClick={key.series_uid ? () => setShow({series_id:key.series_uid, study_name:key.studyName}) : undefined}
                        >
                            {key.pathology} 
                        </td>
                        <td 
                            className="px-6 py-4" 
                            onClick={key.series_uid ? () => setShow({series_id:key.series_uid, study_name:key.studyName}) : undefined}
                        >
                            {key.most_dangerous_pathology_type} 
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
        </>
    )
};
export default StudiesPage;



// export default function StudiesPage(): JSX.Element {
//     const { setShow } = useContext(showImWorkpaceContext) as IShowImWorkspaceContext
//     const [studies, setStudies] = useState<Array<StudyInfo>>([])
    
//     useEffect(() => {
//         getStudies(setStudies)
//         return () => setShow({ href: "", num: 0, comment: false })
//     }, [])

//     // Ставим интервал на проверку
//     useEffect(() => {
//         const interval = setInterval(() => {
//             getStudies(setStudies)
//         }, 30000);
//         return () => clearInterval(interval);
//     }, []);

//     const showStudy = (key: string, num: number, comment: boolean = false) => {
//         if (num !== 0)
//             setShow({ href: key, num: num, comment: comment })
//     }

//     return (
//         <table className="w-full text-sm text-center text-gray-500 table-auto">
//             <thead className="font-bold text-gray-700 uppercase bg-gray-50">
//                 <tr>
//                     <th scope="col" className="px-6 py-3">
//                         Исследование
//                     </th>
//                     <th scope="col" className="px-6 py-3">
//                         Статус обработки
//                     </th>
//                     <th scope="col" className="px-6 py-3">
//                         Наличие патологии
//                     </th>
//                     <th scope="col" className="px-6 py-3">
//                         Где патология
//                     </th>
//                 </tr>
//             </thead>
//             <tbody>
//                 {studies.map((key) => (
//                     <tr 
//                         key={key.series_uid + key.studyName} 
//                         className={`${Number(key.proccessed) !== true ? 'bg-gray-300' : 'bg-red-300'} border-b text-center hover:*:underline cursor-pointer`}
//                     >
//                         <td 
//                             scope="row" 
//                             className="px-6 py-4 font-medium text-gray-900 break-all" 
//                             onClick={() => showStudy(key.StudyID, Number(key.NumStudies), key.SetComments)}
//                         >
//                             {key.study_uid + "/" + key.series_uid} 
//                         </td>
//                         <td 
//                             className="px-6 py-4" 
//                             onClick={() => showStudy(key.StudyID, Number(key.NumStudies), key.SetComments)}
//                         >
//                             {key.processing_status}
//                         </td>
//                         <td 
//                             className="px-6 py-4" 
//                             onClick={() => showStudy(key.StudyID, Number(key.NumStudies), key.SetComments)}
//                         >
//                             {key.pathology} 
//                         </td>
//                         <td 
//                             className="px-6 py-4" 
//                             onClick={() => showStudy(key.StudyID, Number(key.NumStudies), key.SetComments)}
//                         >
//                             {key.most_dangerous_pathology_type} 
//                         </td>
//                     </tr>
//                 ))}
//             </tbody>
//         </table>
//     )
// }