
import { ReactNode } from "react"
import { Link } from "react-router-dom"


export default function LeftPanel():ReactNode{

    const links = [['Исследования в работе','/Studies'],['Загрузить исследование','/Upload'], ['Выход','/logout']]

    return (
    <>

        <aside id="default-sidebar" className={"w-screen sm:fixed sm:w-64 top-0 left-0 z-40  sm:h-screen text-black " } aria-label="Sidebar">
            
            <div className="h-full px-3 py-4 overflow-y-auto bg-white">
                <Logo />
                <ul className="space-y-2 font-medium" >
                    {links.map( (key)=> (
                            <li key={key[1]}>
                                <Link to={key[1]} className="flex items-center text-center p-2 text-gray-900 rounded-lg  hover:bg-gray-100 group">
                                    <span className='flex-1 ms-3 uppercase'>{key[0]}</span>
                                </Link>
                            </li>)
                            )
                        }
                </ul>
            </div>

        </aside>

    </>
    )

}


function Logo():ReactNode{


    return (
    <>
        <a href="/" className="hidden sm:flex flex-col items-center ps-2.5 p-2 mb-5 font-bold text-green-600 text-xl">
            AIRI FM
        </a>

            
    </>
    )

}