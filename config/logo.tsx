import Image from "next/image"
import logo from "@/public/favicon.png"

export function Logo({ className = "h-6 w-6" }: { className?: string }) {
  return (
    <div className={className}>
      <Image src={logo} alt="Logo" width={24} height={24} />
    </div>
  )
}
