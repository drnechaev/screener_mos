import './App.css'
import PrimePage from './pages/PrimePage';
import IsAuth from './auth/auth';


import {BrowserRouter as Router } from 'react-router-dom';


export default function App() {

  if(import.meta.env.MODE=='production')
  {
    console.log = ()=>{}
    console.debug = ()=>{}
  }
  
  console.log("Backend host", import.meta.env.VITE_GUI_BACKEND )

return (

    <Router>
      <IsAuth>
        <PrimePage/>
      </IsAuth>
    </Router>
    
   )
}



