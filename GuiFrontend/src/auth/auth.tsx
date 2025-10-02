import { ReactNode, PropsWithChildren} from "react";
import LoginPage from "./login";
import axios from "axios";
import { AxiosError } from "axios";
import { LogOut } from "./login";


export const setToken = (token:string)=>{

    console.log(token)
    localStorage.setItem('temitope', token)// make up your own token
}

export const fetchToken = ():string =>{

    const token = localStorage.getItem('temitope')
    if(token===null)
        return ""
    return token
}


export function LogOutResponeCheck(error:AxiosError|any):boolean{
    if(error && axios.isAxiosError(error))
    {
        console.error("erros",error.response?.status);
        if(error.response?.status==401)
            LogOut()
    }
    else
        console.error(error)

    return false

}

export default function IsAuth({children}:PropsWithChildren):ReactNode{

    //let location = useLocation()
    console.log("IsAuth")
    if(fetchToken()!=="")
    {
        axios.defaults.headers.common['Authorization'] = `Bearer ${fetchToken()}`
        return children
    }
    else
        return <LoginPage/>

}