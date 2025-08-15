import Link from "next/link";

const NavLink = ({
  href,
  title,
  onClick,
  className,
}: {
  href: string;
  title: string;
  onClick?: () => void;
  className?: string;
}) => {
  return (
    <Link
      href={href}
      onClick={onClick}
      className={`block py-2 pl-3 pr-4 text-[#ADB7BE] sm:text-xl rounded md:p-0 hover:text-white ${className}`}
    >
      {title}
    </Link>
  );
};

export default NavLink;
