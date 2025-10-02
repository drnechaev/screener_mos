
import { ChangeEvent, FormEvent, ReactNode, useState} from "react";
import { setToken } from "./auth";
import axios from "axios";

type AuthValue = {
    sc_login:string
    sc_password:string
}

async function Autorize(vals:AuthValue,  e:FormEvent<HTMLFormElement>) {
        
    e.preventDefault()


    const form_data = new FormData()

    console.log(import.meta.env.MODE)

    if(import.meta.env.MODE!=='development')
    {

        form_data.append("username", vals.sc_login)
        form_data.append("password", vals.sc_password)
        console.log(form_data)
        
        try {
            console.log("Send auth query")
            const response = await axios.post(`${import.meta.env.VITE_GUI_BACKEND}/user/token`, form_data);
            console.log(response); //Will result in an error because the above endpoint doesn't exist yet
            if(response.data.access_token){
                console.log(response)
                setToken(response.data.access_token)
                window.location.reload()
            }
        }catch (error) {
            console.error("erros",error);
        }
    }
    else
    {
        console.log("Send auth query")
        form_data.append("username", 'admin')
        form_data.append("password", '123')
        console.log(form_data)
        
        try {
            console.log("Send auth query", `${import.meta.env.VITE_GUI_BACKEND}/user/token`)
            const response = await axios.post(`${import.meta.env.VITE_GUI_BACKEND}/user/token`, form_data);
            console.log(response); //Will result in an error because the above endpoint doesn't exist yet
            if(response.data.access_token){
                console.log(response)
                setToken(response.data.access_token)
                window.location.reload()
            }
        }catch (error) {
            console.error("erros",error);
        }
        window.location.reload()
    }
}

export default function LoginPage():ReactNode{
    
    console.log("Login page")
    const [AuthData, SetAuthData] = useState<AuthValue>({sc_login:"",sc_password:""})


    const handleChangeValue = (e:ChangeEvent<HTMLInputElement>) => {
        //  меняем значения формы поля
        if(!e.target)
            return;

        const {name, value} = e.target
        SetAuthData({...AuthData,[name]:value})
    
    }
    

    return ( 
        <div className="flex w-screen h-screen items-center justify-center bg-blue-200">

            <div className=" p-10 pt-0 bg-white rounded-xl text-black ">
                <h1 className="text-xl font-bold p-3 ">AIRI FM Service</h1>
                <h3 className="text-lg font-bold p-3 pb-5">Пройдите аутентификацию</h3>
                <form className="flex flex-col items-start space-y-3" onSubmit={(e) => Autorize(AuthData, e)}>
                    <input name="sc_login" type='text' placeholder="Логин" required={import.meta.env.MODE!='development'?true:false} onChange={handleChangeValue} className="bg-gray-50 border-2 border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:outline-none focus:border-blue-500 block w-full p-2.5 "/>
                    <input name="sc_password" type="password"  placeholder="Пароль" required={import.meta.env.MODE!='development'?true:false} onChange={handleChangeValue}  className="border-2 bg-gray-50 border-gray-300 p-2.5 w-full text-sm selection:text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 focus:outline-none"/>
                    <button type="submit" className="text-white p-3 bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm w-full">
                        Авторизироваться
                    </button>
                </form>
            </div>

        </div>
    )

}

  

export function LogOut():ReactNode{

    console.log("Log out")
    setToken("")  
    window.location.href = '/'

    return <></>

}